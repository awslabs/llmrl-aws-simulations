import copy
import random
import string
import numpy as np
from typing import List, Optional, Dict
import concurrent.futures
from prompt.prompt_templates import gpt_template
import json
import shlex
import orjson
import os
import atexit

from envs.account_utils import (
    PREFIX,
    S3_ACL_OPTIONS,
    S3_OBJECT_OWNERSHIP_OPTIONS,
    STORAGE_LIMITS,
    AMI_IDS,
    EC2_STATES,
    EC2_INSTANCE_TYPES,
    RDS_INSTANCE_TYPES,
    RDS_ENGINES,
    RDS_STORAGE_TYPES,
    MotoServer,
    HiddenEnvState,
    Query,
    query_combo_list,
    run_cli_command,
    sentence_template,
)


class ChatHistory:
    def __init__(
        self, add_type_of_content: bool = False, condense_chat_text: bool = True
    ):
        self._chats = []
        self.chat_text = ""
        self.condense_chat_text = condense_chat_text
        self.add_type_of_content = add_type_of_content

    def add_chat(self, response: str, role: str = "user", prompt: Optional[str] = None):
        if prompt is not None:
            assert role == "user", "only users can pass the prompt"
            content = {"text": f"Prompt: {prompt}" + "\n " + response}
        else:
            content = {"text": response}
        if self.add_type_of_content:
            content["type"] = "text"
        chat = {
            "role": role,
            "content": [content],
        }
        self._chats.append(chat)
        if self.condense_chat_text:
            self.chat_text = self.chat_text + f" {role}: {response}"
        else:
            self.chat_text = self.chat_text + " " + str(chat)

    def history(
        self,
        return_text: bool = False,
    ):
        if return_text:
            return copy.deepcopy(self.chat_text)
        else:
            return copy.deepcopy(self._chats)

    def clear(self):
        self._chats = []
        self.chat_text = ""


class AWSEnv:
    regions = [
        "us-east-1",  # Focusing only on 1 region for simplicity
    ]

    DEFAULT_PROMPT: str = gpt_template()
    s3_acl_options: List = S3_ACL_OPTIONS
    s3_object_ownership_options: List = S3_OBJECT_OWNERSHIP_OPTIONS
    storage_limits: List = STORAGE_LIMITS
    ami_ids: List = AMI_IDS
    ec2_state: List = EC2_STATES
    ec2_instance_types: List = EC2_INSTANCE_TYPES
    rds_instance_types: List = RDS_INSTANCE_TYPES
    rds_storage_types: List = RDS_STORAGE_TYPES
    rds_engines: List = RDS_ENGINES
    max_reward: float = 100.0
    correct_command_reward: float = 0.1

    def __init__(
        self,
        seed: int = 0,
        env_id: int = 0,
        port_number: Optional[int] = None,
        max_num_buckets: int = 5,
        max_num_rds: int = 5,
        max_num_roles: int = 5,
        max_num_ec2s: int = 5,
        min_num_buckets: int = 1,
        min_num_rds: int = 1,
        min_num_roles: int = 1,
        min_num_ec2s: int = 1,
        max_conversation_length: int = 10,
        reset_user_upon_done: bool = False,
        clear_chat_history_per_user_query: bool = True,
        return_state_as_text: bool = True,
        use_default_policy_names: bool = False,
        add_type_of_content: bool = False,
        condense_chat_text: bool = True,
        existing_prefix_sampling_prob: float = 0.9,
        service_sampling_epsilon: float = 0.05,
        log_path: str = "./",
        log_every: int = -1,
        env_type: str = "easy",
        *args,
        **kwargs,
    ):
        assert env_type in ["easy", "medium", "hard"]
        if env_type == "easy":
            use_default_query_text = True
        else:
            use_default_query_text = False
        self.env_type = env_type

        self.env_id = env_id
        seed = seed + env_id
        if port_number is None:
            port_number = seed
        # have a different port for each env
        port_number = port_number + env_id
        self.seed = seed
        self.service_span = {}
        assert min_num_buckets <= max_num_buckets
        self.max_num_buckets = max_num_buckets
        self.min_num_buckets = min_num_buckets
        self.service_span["s3"] = self.max_num_buckets - self.min_num_buckets
        assert min_num_rds <= max_num_rds
        self.max_num_rds = max_num_rds
        self.min_num_rds = min_num_rds
        self.service_span["rds"] = self.max_num_rds - self.min_num_rds
        assert min_num_roles <= max_num_roles
        self.max_num_roles = max_num_roles
        self.min_num_roles = min_num_roles
        self.service_span["iam"] = self.max_num_roles - self.min_num_roles
        assert min_num_ec2s <= max_num_ec2s
        self.max_num_ec2s = max_num_ec2s
        self.min_num_ec2s = min_num_ec2s
        self.service_span["ec2"] = self.max_num_ec2s - self.min_num_ec2s

        self.use_default_policy_names = use_default_policy_names
        self.hidden_state = HiddenEnvState()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.server = MotoServer(port=5000 + port_number)
        self.max_conversation_length = max_conversation_length
        self._steps = 0
        self.done = False
        self.reset_user_upon_done = reset_user_upon_done
        self.clear_chat_history_per_user_query = clear_chat_history_per_user_query
        self._user_queries = query_combo_list(query_cls=self.env_type)
        self.return_state_as_text = return_state_as_text
        self.chat_history = ChatHistory(
            add_type_of_content=add_type_of_content,
            condense_chat_text=condense_chat_text,
        )
        self.existing_prefix_sampling_prob = existing_prefix_sampling_prob
        self.service_sampling_epsilon = service_sampling_epsilon
        self._info = {
            "num queries asked": 0,
            "num queries answered": 0,
            "total successes": 0,
            "queries with zero as answer": 0,
            "total reward": 0,
            "num correct CLI commands": 0,
            "num total CLI attempts": 0,
            "query_types": [],
            # "env_id": [],
            "num_queries_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "num_answered_queries_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "correct_answered_queries_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "correct_commands_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "incorrect_commands_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
        }
        self.closed = True
        self._logs = {"interactions": []}
        self._log_every = log_every
        self._num_episodes = 0
        log_path = os.path.join(log_path, f"env_id_{self.server.port}")
        os.makedirs(log_path, exist_ok=True)
        self._log_path = log_path
        self._sentence_template = sentence_template()
        self.use_default_query_text = use_default_query_text
        atexit.register(self.server.kill)

    def close(self):
        # tear down the server
        self.closed = True
        self.server.kill()

    def reset_info(self):
        self._info = {
            "num queries asked": 0,
            "num queries answered": 0,
            "total successes": 0,
            "queries with zero as answer": 0,
            "total reward": 0,
            "num correct CLI commands": 0,
            "num total CLI attempts": 0,
            "query_types": [],
            # "env_id": [],
            "num_queries_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "num_answered_queries_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "correct_answered_queries_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "correct_commands_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
            "incorrect_commands_per_service": {
                "s3": 0,
                "rds": 0,
                "iam": 0,
                "ec2": 0,
            },
        }

    def reset_logs(self):
        self._logs = {"interactions": []}

    @property
    def region(self):
        return self._region

    def sample_user_query(self):
        # sample a query_type
        query_combo = self.rng.choice(self._user_queries)
        # for each possible query parameters sample one
        prob_per_service = []
        total_services = 0
        # only look at the services within the query combo
        for service in query_combo.services:
            num_services = len(self.hidden_state.get_service(service).ids)
            prob_per_service.append(num_services)
            total_services += num_services
        # if we have at least one of the service, sample proportionally to their occurence
        if total_services > 0:
            prob_per_service = [
                (1 - self.service_sampling_epsilon) * p / total_services
                + self.service_sampling_epsilon * 1 / len(prob_per_service)
                for p in prob_per_service
            ]
            service = self.rng.choices(
                query_combo.services, weights=prob_per_service, k=1
            )[0]
        else:
            service = self.rng.choice(query_combo.services)
        service_properties = query_combo.service_properties[service]["service_options"]
        query_options = self.rng.choice(service_properties)
        param = self.rng.choice(query_options[1])

        queried_property = query_options[2]
        query_type = query_options[-1]

        if "prefix" in query_type:
            # resampling with bias towards picking prefixes that are in the accounts.
            ids = self.hidden_state.get_service(service).ids
            if len(ids) > 0:
                if param not in ids:
                    # all prefix are of the same length
                    existing_prefix = self.rng.choice(ids)[: len(param)]
                    param = self.rng.choices(
                        [param, existing_prefix],
                        weights=[
                            1 - self.existing_prefix_sampling_prob,
                            self.existing_prefix_sampling_prob,
                        ],
                        k=1,
                    )[0]
        query_template = self._sentence_template[query_type]
        # sample a random query template from the list of templates
        query_sentence = self.rng.choices(query_template["templates"])[0]
        # replace the service flag
        query_sentence = query_sentence.replace(
            "[SERVICE]", query_combo.service_properties[service]["service_name"]
        )
        # replace the service name we are querying about
        if isinstance(query_template["name"][service], Dict):
            sentence_name = query_template["name"][service][queried_property]
        else:
            sentence_name = query_template["name"][service]
        query_sentence = query_sentence.replace("[NAME]", sentence_name)
        if param == "":
            query_end = query_options[0] + "?"
        else:
            query_sentence = query_sentence.replace("[VAL]", param)
            query_end = query_options[0] + " " + param + "?"

        if query_type.startswith("check_specific_if"):
            # sample a random service id (instance id, role name, bucket name etc) for this specific service.
            if len(self.hidden_state.get_service(service).ids) > 0:
                service_id = self.rng.choice(self.hidden_state.get_service(service).ids)
            else:
                service_id = self.get_random_string()
            query_text = (
                query_combo.query_start
                + " "
                + query_combo.service_properties[service]["service_name"]
                + " "
                + service_id
                + " "
                + query_end
            )
            # replace the service ID with the one we are querying about
            query_sentence = query_sentence.replace("[SERVICE_ID]", service_id)
        else:
            query_text = (
                query_combo.query_start
                + " "
                + query_combo.service_properties[service]["service_name"]
                + " "
                + query_end
            )
            service_id = None
        query = Query(
            query_text=query_text if self.use_default_query_text else query_sentence,
            service=service,
            param=param,
            query_type=query_type,
            service_id=service_id,
            queried_property=queried_property,
        )

        self._current_query = query
        # for the given query, get the ground truth response --> Currently this function is hard-coded.
        # But ideally, the Amazon env should store the ground truth response.
        self._ground_truth_response = self.get_ground_truth_response(query)

    def guess_penalty(self):
        # penalty in expectation is -10
        if "list_and_count" in self._current_query.query_type:
            # get how many of elements we have at most per service.
            frequency = self.service_span[self._current_query.service]
            # avoid division with 0
            frequency = max(frequency - 1, 1)
            # probability of random guessing, getting it right is 1 / num_total_services
            # penalize it for getting it wrong.
            return self.max_reward * 1.0 / frequency + self.max_conversation_length
        elif "check_if_any" in self._current_query.query_type:
            # 50 % chance to get it right
            return self.max_reward + self.max_conversation_length
        elif "check_specific_if" in self._current_query.query_type:
            # 50 % chance to get it right
            return self.max_reward + self.max_conversation_length
        else:
            raise NotImplementedError

    def get_ground_truth_response(self, query: Query) -> int:
        if query.query_type == "list_and_count":
            service = query.service
            service = self.hidden_state.get_service(service_type=service)
            return service.num_items
        elif (
            query.query_type == "list_and_count_by_prefix"
            or query.query_type == "check_if_any_by_prefix"
        ):
            service = query.service
            service = self.hidden_state.get_service(service_type=service)
            prefix = query.param
            if query.query_type == "check_if_any_by_prefix":
                return int(service.num_items_with_prefix(prefix) > 0)
            else:
                return service.num_items_with_prefix(prefix)
        elif (
            query.query_type == "list_and_count_by_kwarg_enabled"
            or query.query_type == "check_if_any_by_kwarg_enabled"
        ):
            service = query.service
            assert service != "s3"
            service = self.hidden_state.get_service(service_type=service)
            if query.query_type == "check_if_any_by_kwarg_enabled":
                return int(service.num_items_with_kwarg_enabled(query.param) > 0)
            else:
                return service.num_items_with_kwarg_enabled(query.param)
        elif (
            query.query_type == "list_and_count_by_storage"
            or query.query_type == "check_if_any_by_storage"
        ):
            service = query.service
            assert service == "rds"
            service = self.hidden_state.get_service(service_type=service)
            if query.query_type == "check_if_any_by_storage":
                return int(service.num_items_with_storage(query.param) > 0)
            else:
                return service.num_items_with_storage(query.param)
        elif query.query_type == "check_specific_if_kwarg_enabled":
            service = query.service
            assert service != "s3"
            service = self.hidden_state.get_service(service_type=service)
            return int(
                service.item_with_kwarg_enabled(
                    query.param, service_id=query.service_id
                )
            )
        elif query.query_type == "check_specific_if_storage":
            service = query.service
            assert service == "rds"
            service = self.hidden_state.get_service(service_type=service)
            return int(
                service.item_with_storage(query.param, service_id=query.service_id)
            )
        elif (
            query.query_type == "list_and_count_by_property"
            or query.query_type == "check_if_any_by_property"
        ):
            queried_property = query.queried_property
            service = query.service
            service = self.hidden_state.get_service(service_type=service)
            desired_property = query.param
            num_items = service.num_items_with_property(
                property=queried_property, desired_property=desired_property
            )
            if query.query_type == "check_if_any_by_property":
                return int(num_items > 0)
            else:
                return num_items
        elif query.query_type == "check_specific_if_property":
            service = query.service
            service = self.hidden_state.get_service(service_type=service)
            queried_property = query.queried_property
            desired_property = query.param
            return int(
                service.item_with_property(
                    property=queried_property,
                    desired_property=desired_property,
                    service_id=query.service_id,
                )
            )
        else:
            raise NotImplementedError

    def get_random_string(self, length: int = 8):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = "".join(self.rng.choice(letters) for i in range(length))
        prefix = self.rng.choice(PREFIX)
        result_str = prefix + "-" + result_str
        return result_str

    def create_s3_buckets(self):
        # TODO: Check if this function is correct --> Do we add the properties that we query about correctly?
        num_s3_buckets = self.np_rng.integers(
            low=self.min_num_buckets, high=self.max_num_buckets + 1
        )

        def create_bucket(bucket_name: str, acl: str, object_ownership: str):
            command = copy.deepcopy(self.base_command)
            command.extend(["s3api", "create-bucket",
                            "--bucket", bucket_name,
                            "--acl", acl,
                            "--object-ownership", object_ownership])
            self.hidden_state.add_to_state(
                service_type="s3",
                bucket_name=bucket_name,
                acl=acl,
                object_ownership=object_ownership,
            )
            result = run_cli_command(command=command)
            return True

        for bucket in range(num_s3_buckets):
            bucket_name = self.get_random_string()
            bucket_name += "-s3"
            acl = self.rng.choice(self.s3_acl_options)
            object_ownership = self.rng.choice(self.s3_object_ownership_options)
            create_bucket(bucket_name, acl, object_ownership)

    def create_rds_instances(self):
        # TODO: Check if this function is correct --> Do we add the properties that we query about correctly?
        def create_rds(
            instance_name: str,
            storage: int,
            public: bool,
            instance_type: str,
            storage_type: str,
            db_engine: str,
            encrypted: bool,
        ):
            command = copy.deepcopy(self.base_command)
            self.hidden_state.add_to_state(
                service_type="rds",
                instance_name=instance_name,
                storage=storage,
                public_access=public,
                instance_type=instance_type,
                storage_type=storage_type,
                db_engine=db_engine,
                encryption_enabled=encrypted,
            )
            command.extend(
                [
                    "rds",
                    "create-db-instance",
                    "--region",
                    self._region,
                    "--db-instance-identifier",
                    instance_name,
                    "--db-instance-class",
                    instance_type,
                    "--engine",
                    "mysql",
                    "--master-username",
                    "admin",
                    "--master-user-password",
                    "mypassword123",
                    "--allocated-storage",
                    f"{storage}",
                    "--engine",
                    db_engine,
                    "--storage-type",
                    storage_type,
                ]
            )
            # TODO: Check if this flag works
            if public:
                command.extend(["--publicly-accessible"])
            else:
                command.extend(["--no-publicly-accessible"])
            if encrypted:
                command.extend(["--storage-encrypted"])
            result = run_cli_command(command=command)
            return True

        num_rds = self.np_rng.integers(low=self.min_num_rds, high=self.max_num_rds + 1)
        for rds in range(num_rds):
            storage = self.rng.choice(self.storage_limits)
            public = self.rng.choice([True, False])
            instance_name = self.get_random_string()
            instance_name += "-rds"
            instance_type = self.rng.choice(self.rds_instance_types)
            storage_type = self.rng.choice(self.rds_storage_types)
            db_engine = self.rng.choice(self.rds_engines)
            encrypted = self.rng.choice([True, False])
            create_rds(instance_name=instance_name,
                       storage=storage,
                       public=public,
                       instance_type=instance_type,
                       storage_type=storage_type,
                       db_engine=db_engine,
                       encrypted=encrypted)

    def create_iam_roles(self):
        def create_role(
            role_name: str,
            user_name: str,
            ec2_access: bool,
            s3_access: bool,
            rds_access: bool,
        ):
            command = copy.deepcopy(self.base_command)
            self.hidden_state.add_to_state(
                service_type="iam",
                role_name=role_name,
                user_name=user_name,
                ec2_access=ec2_access,
                s3_access=s3_access,
                rds_access=rds_access,
            )

            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "AWS": user_name,
                        },
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            command.extend(
                [
                    "iam",
                    "create-role",
                    "--region",
                    self._region,
                    "--role-name",
                    role_name,
                    "--assume-role-policy-document",
                    json.dumps(trust_policy),
                ]
            )
            result = run_cli_command(command=command)
            # add access to different services for the role
            if ec2_access:
                command = copy.deepcopy(self.base_command)
                ec2_document = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {"Effect": "Allow", "Action": "ec2:*", "Resource": "*"}
                    ],
                }
                command.extend(
                    [
                        "iam",
                        "put-role-policy",
                        "--region",
                        self._region,
                        "--role-name",
                        role_name,
                        "--policy-name",
                        (
                            "EC2Access"
                            if self.use_default_policy_names
                            else self.get_random_string()
                        ),
                        "--policy-document",
                        json.dumps(ec2_document),
                    ]
                )
                result = run_cli_command(command=command)
            if s3_access:
                command = copy.deepcopy(self.base_command)
                s3_document = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {"Effect": "Allow", "Action": "s3:*", "Resource": "*"}
                    ],
                }
                command.extend(
                    [
                        "iam",
                        "put-role-policy",
                        "--region",
                        self._region,
                        "--role-name",
                        role_name,
                        "--policy-name",
                        (
                            "S3Access"
                            if self.use_default_policy_names
                            else self.get_random_string()
                        ),
                        "--policy-document",
                        json.dumps(s3_document),
                    ]
                )
                result = run_cli_command(command=command)
            if rds_access:
                command = copy.deepcopy(self.base_command)
                rds_document = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {"Effect": "Allow", "Action": "rds:*", "Resource": "*"}
                    ],
                }
                command.extend(
                    [
                        "iam",
                        "put-role-policy",
                        "--region",
                        self._region,
                        "--role-name",
                        role_name,
                        "--policy-name",
                        (
                            "RDSAccess"
                            if self.use_default_policy_names
                            else self.get_random_string()
                        ),
                        "--policy-document",
                        json.dumps(rds_document),
                    ]
                )
                result = run_cli_command(command=command)
            return True

        num_roles = self.np_rng.integers(
            low=self.min_num_roles, high=self.max_num_roles + 1
        )
        for role in range(num_roles):
            ec2_access = self.rng.choice([True, False])
            s3_access = self.rng.choice([True, False])
            rds_access = self.rng.choice([True, False])
            role_name = self.get_random_string()
            role_name += "-role"
            user_name = self.get_random_string()
            user_name += "-user"
            create_role(
                role_name=role_name,
                user_name=user_name,
                ec2_access=ec2_access,
                s3_access=s3_access,
                rds_access=rds_access,
            )

    def create_ec2_instances(self):
        def create_ec2_instance(
            ami_id: str,
            public: bool,
            ec2_states: List[str],
            instance_type: str,
            monitoring_enabled: bool,
            count: int = 1,
        ):
            command = copy.deepcopy(self.base_command)
            command.extend(
                [
                    "ec2",
                    "run-instances",
                    "--region",
                    self._region,
                    "--image-id",
                    f"ami-{ami_id}",
                    "--count",
                    str(count),
                    "--instance-type",
                    instance_type,
                    "--security-group-id",
                    self.server.sg,
                    "--subnet-id",
                    self.server.subnet_id,
                ]
            )
            # TODO: Check if this works even though we might have a public/private subnet
            if public:
                command.extend(["--associate-public-ip-address"])
            else:
                command.extend(["--no-associate-public-ip-address"])
            if monitoring_enabled:
                command.extend(["--monitoring", "Enabled=true"])
            else:
                command.extend(["--monitoring", "Enabled=false"])
            user_data = f"""#!/bin/bash
                    echo "Instance Unique ID: 1234" > /tmp/instance_info.txt
                    """
            command.extend(["--user-data", user_data])
            result = run_cli_command(command=command)
            try:
                response = json.loads(result.stdout)
            except json.decoder.JSONDecodeError:
                raise AssertionError(
                    f"received error {result.stderr} for command {command}"
                )
            for i, instance in enumerate(response["Instances"]):
                instance_id = instance["InstanceId"]
                ec2_state = ec2_states[i]
                if ec2_state == "running":
                    command = copy.deepcopy(self.base_command)
                    command.extend(
                        [
                            "ec2",
                            "start-instances",
                            "--region",
                            self._region,
                            "--instance-id",
                            instance_id,
                        ]
                    )
                elif ec2_state == "stopped":
                    command = copy.deepcopy(self.base_command)
                    command.extend(
                        [
                            "ec2",
                            "stop-instances",
                            "--region",
                            self._region,
                            "--instance-id",
                            instance_id,
                        ]
                    )
                elif ec2_state == "terminated":
                    command = copy.deepcopy(self.base_command)
                    command.extend(
                        [
                            "ec2",
                            "terminate-instances",
                            "--region",
                            self._region,
                            "--instance-id",
                            instance_id,
                        ]
                    )
                else:
                    raise AssertionError(f"State {ec2_state} not supported")
                result = run_cli_command(command=command)
                self.hidden_state.add_to_state(
                    instance_id=instance_id,
                    ami_id=f"ami-{ami_id}",
                    service_type="ec2",
                    assign_public_ip=public,
                    state=ec2_state,
                    instance_type=instance_type,
                    monitoring_enabled=monitoring_enabled,
                )
            return True

        total_instances = self.np_rng.integers(
            low=self.min_num_ec2s, high=self.max_num_ec2s + 1
        )
        instances_sampled = 0
        while instances_sampled < total_instances:
            public = self.rng.choice([True, False])
            monitoring_enabled = self.rng.choice([True, False])
            ami_id = self.rng.choice(self.ami_ids)
            instance_type = self.rng.choice(self.ec2_instance_types)
            count = self.np_rng.integers(low=1, high=3)
            ec2_states = self.rng.choices(self.ec2_state, k=count)
            instances_sampled += count
            create_ec2_instance(
                ami_id=ami_id,
                public=public,
                count=count,
                ec2_states=ec2_states,
                instance_type=instance_type,
                monitoring_enabled=monitoring_enabled,
            )

    def reset(self):
        self.closed = False
        self.server.restart()
        self.hidden_state = HiddenEnvState()
        self._region = self.rng.choice(self.regions)
        self.base_command = [
            "aws",
            "--endpoint-url",
            self.server.endpoint,
        ]

        # create s3 buckets
        self.create_s3_buckets()

        # create rds instances
        self.create_rds_instances()

        # create IAM roles
        self.create_iam_roles()

        # create EC2 instances
        self.create_ec2_instances()

        # reset info/logging metrics
        self.reset_info()

        # once the state of the AWS user is set, reset to a new user query
        self.chat_history.clear()
        self.reset_user_query()
        self.reset_logs()

        self._steps = 0
        self.done = False
        self._num_episodes += 1
        return self.state

    @staticmethod
    def format_user_response(
        query_text: str = "", system_output: str = "", system_error: str = ""
    ) -> str:
        response = ""
        if query_text != "":
            response += f'"Query": {query_text}\n'
        if system_output != "":
            response += f'"SystemOutput": {system_output}\n'
        if system_error != "":
            response += f'"SystemError": {system_error}\n'
        # if we get no response from the server or user, return empty output.
        if query_text == "" and system_output == "" and system_error == "":
            # TODO: Check if this logic is valid
            response += '"SystemOutput": []\n'
        return response

    @staticmethod
    def format_model_output(input: str = "", answer: str = ""):
        response = ""
        if input != "":
            response += f'"Input": {input}\n'
        if answer != "":
            response += f'"Answer": {answer}\n'
        return response

    def reset_user_query(self):
        if self.clear_chat_history_per_user_query:
            self.chat_history.clear()
        self.sample_user_query()
        # query_text = self.DEFAULT_PROMPT + "\n" + self._current_query.query_text
        # self.chat_history.add_chat(response=self.DEFAULT_PROMPT, role='user')
        query = self.format_user_response(
            query_text=self._current_query.query_text,
        )
        self.chat_history.add_chat(
            response=query, role="user", prompt=self.DEFAULT_PROMPT
        )
        # reset step and done counters
        self._info["num queries asked"] += 1
        self._info["queries with zero as answer"] += self._ground_truth_response == 0
        self._info["query_types"].append(copy.deepcopy(self._current_query.query_type))
        self._info["num_queries_per_service"][self._current_query.service] += 1
        # self._info["env_id"].append(copy.deepcopy(self.server.port))

    @staticmethod
    def convert_sys_response(system_response):
        # try compressing the outut using orjson. If it doesn't work, return the response as is.
        try:
            qwe = orjson.loads(system_response)
            resp = orjson.dumps(qwe).decode("utf-8")
            return resp
        except ValueError:
            return system_response

    def step(self, action: str):
        assert not self.closed, "The server is closed, call reset function before step"
        self._steps += 1
        step_data = {
            "step": copy.deepcopy(self._steps),
            "observation": self.state,
            "action": action,
        }
        execute_command = copy.deepcopy(self.base_command)
        execute_command.extend(["--region", self._region])
        # By default we give a reward of -1
        reward = -1
        done = False
        if self.done:
            return None
        input_command = action
        answer = ""
        # Checking if the agent provides the action in the correct syntax
        if action.startswith('"Input":'):
            # remove the brackets and aws command
            input_command = action[len('"Input":') :]
            splitted_action = input_command.split("aws")[-1]
            # CLI input attempted
            self._info["num total CLI attempts"] += 1
            try:
                splitted_action = shlex.split(splitted_action)
                execute_command.extend(splitted_action)
                out = run_cli_command(execute_command)
                system_output = self.convert_sys_response(out.stdout)
                system_error = self.convert_sys_response(out.stderr)
                # if we receive no system error, correct CLI was executed
                if system_error == "":
                    self._info["num correct CLI commands"] += 1
                    # add to the service for which the correct command was executed
                    self._info["correct_commands_per_service"][
                        self._current_query.service
                    ] += 1
                    reward += self.correct_command_reward
                else:
                    self._info["incorrect_commands_per_service"][
                        self._current_query.service
                    ] += 1
            except ValueError:
                system_output = ""
                system_error = "Invalid input"
                self._info["incorrect_commands_per_service"][
                    self._current_query.service
                ] += 1

            # splitted_action = splitted_action.split(" ")
            # if splitted_action[0] == "":
            #    splitted_action = splitted_action[1:]
            # execute the cli command

        elif action.startswith('"Answer":'):
            answer = action[len('"Answer":') :]
            try:
                ans_to_int = int(answer)
                self._info["num queries answered"] += 1
                self._info["num_answered_queries_per_service"][
                    self._current_query.service
                ] += 1
                # If the answer is the same as the ground_truth. We can return True.
                if ans_to_int == self._ground_truth_response:
                    system_output = f"Congratulations! You got the correct answer. {self._ground_truth_response}"
                    system_error = ""
                    # if the agent guesses the right answer, we give a reward of 100 and set the done flag to True.
                    reward = self.max_reward
                    done = True
                    self._info["total successes"] += 1
                    self._info["correct_answered_queries_per_service"][
                        self._current_query.service
                    ] += 1
                else:
                    system_output = (
                        f"Unfortunately, your answer is incorrect. "
                        f"The answer was {self._ground_truth_response}."
                    )
                    system_error = ""
                    penalty = self.guess_penalty()
                    reward += -penalty
                    done = True
            except:
                system_output = ""
                system_error = "Answer must be an integer."
        else:
            system_output = ""
            system_error = 'Command has to start with either "Input": or "Answer": '
            # if output was completely wrong, we consider it as a wrong CLI command
            self._info["num total CLI attempts"] += 1
            self._info["incorrect_commands_per_service"][
                self._current_query.service
            ] += 1

        action_summary = self.format_model_output(input=input_command, answer=answer)
        self.chat_history.add_chat(response=action_summary, role="assistant")
        system_summary = self.format_user_response(
            system_output=system_output, system_error=system_error
        )
        self.chat_history.add_chat(response=system_summary, role="user")
        # summary = self.state + ",\n" + action_summary
        if done and self.reset_user_upon_done:
            self.reset_user_query()
            done = False
        # if we exceed the maximum number of steps for the conversation, we force a reset

        done = self._steps >= self.max_conversation_length or done
        self.done = done
        self._info["total reward"] += reward
        step_data.update(
            {
                "next observation": self.state,
                "reward": reward,
                "done": done,
            }
        )
        self._logs["interactions"].append(step_data)
        if self._log_every > 0 and self._num_episodes % self._log_every == 0:
            self.log()
        return self.state, reward, done

    @property
    def state(self):
        if self.return_state_as_text:
            return self.chat_history.history(return_text=True)
        else:
            return self.chat_history.history(return_text=False)

    def get_state(self, return_text: bool = True):
        return self.chat_history.history(return_text=return_text)

    @property
    def info(self):
        return copy.deepcopy(self._info)

    def log(self):
        filename = f"episode_{self._num_episodes}.json"
        filepath = os.path.join(self._log_path, filename)
        with open(filepath, "w") as f:
            json.dump(self._logs, f, indent=2)


class BatchedAWSEnv:
    def __init__(
        self,
        seed: int = 0,
        num_envs: int = 2,
        env_kwargs: Optional[Dict] = None,
        use_multiprocessing: bool = True,
    ):
        if env_kwargs is None:
            env_kwargs = {}
        self.num_envs = num_envs
        self.seed = seed
        self.use_multiprocessing = use_multiprocessing
        if self.use_multiprocessing:
            constructor = lambda index: AWSEnv(seed=seed, env_id=index, **env_kwargs)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [executor.submit(constructor, i) for i in range(num_envs)]
                self.env_list = [job.result() for job in jobs]
        else:
            self.env_list = [
                AWSEnv(seed=seed, env_id=i, **env_kwargs) for i in range(num_envs)
            ]

    def reset(self, *args, **kwargs):
        if self.use_multiprocessing:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [executor.submit(env.reset) for env in self.env_list]
                results = [job.result() for job in jobs]
        else:
            results = [env.reset() for env in self.env_list]
        return results

    def step(self, actions_list: List[str]):
        if self.use_multiprocessing:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [
                    executor.submit(env.step, action)
                    for env, action in zip(self.env_list, actions_list)
                ]
                results = [job.result() for job in jobs]
        else:
            results = [
                env.step(action) for env, action in zip(self.env_list, actions_list)
            ]
        return results

    def get_state(self, return_text: bool = True):
        if self.use_multiprocessing:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [
                    executor.submit(env.get_state, return_text=return_text)
                    for env in self.env_list
                ]
                results = [job.result() for job in jobs]
        else:
            results = [env.get_state(return_text=return_text) for env in self.env_list]
        return results

    def info(self):
        info = self.env_list[0].info
        for env in self.env_list[1:]:
            env_info = env.info
            self._merge_info(info_dict_1=info, info_dict_2=env_info)
        return info

    def _merge_info(self, info_dict_1: Dict, info_dict_2: Dict):
        assert info_dict_1.keys() == info_dict_2.keys()
        for key in info_dict_1.keys():
            if isinstance(info_dict_1[key], float) or isinstance(info_dict_1[key], int):
                assert isinstance(info_dict_2[key], float) or isinstance(
                    info_dict_2[key], int
                )
                info_dict_1[key] = info_dict_1[key] + info_dict_2[key]
            elif isinstance(info_dict_1[key], List):
                assert isinstance(info_dict_2[key], List)
                info_dict_1[key].extend(info_dict_2[key])
            elif isinstance(info_dict_1[key], Dict):
                assert isinstance(info_dict_2[key], Dict)
                self._merge_info(info_dict_1[key], info_dict_2[key])
            else:
                raise AssertionError
        return info_dict_1

    @property
    def bsize(self):
        return self.num_envs

    def close(self):
        if self.use_multiprocessing:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [executor.submit(env.close) for env in self.env_list]
                results = [job.result() for job in jobs]
        else:
            results = [env.close() for env in self.env_list]
