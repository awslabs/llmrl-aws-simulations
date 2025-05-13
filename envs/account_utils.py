from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import copy
import subprocess
import json
import os
import requests

PREFIX = ["abc", "def", "ghi", "jkl", "mno", "pqr", "tuv", "wxyz"]

STORAGE_LIMITS = [5, 10, 20, 50, 100]

AMI_IDS = ["123456", "234567", "345678", "456789", "567890"]

EC2_STATES = ["running", "stopped", "terminated"]

S3_ACL_OPTIONS = ["private", "public-read", "authenticated-read"]

S3_OBJECT_OWNERSHIP_OPTIONS = ["BucketOwnerPreferred", "BucketOwnerEnforced"]

EC2_INSTANCE_TYPES = [
    "t2.micro",
    "t2.small",
    "t2.medium",
    "t2.large",
    "m5.large",
    "m5.xlarge",
    "m5.2xlarge",
    "c8g.medium",
    "c8g.large",
    "c8g.xlarge",
    "p4d.24xlarge",
    "g4dn.xlarge",
    "g4dn.2xlarge",
    "g4dn.metal",
    "i8g.large",
    "i8g.xlarge",
    "i8g.2xlarge",
]


RDS_INSTANCE_TYPES = [
    "db.t2.micro",
    "db.t2.small",
    "db.t2.medium",
    "db.t2.large",
    "db.t3.micro",
    "db.t3.small",
    "db.t3.medium",
    "db.t3.large",
    "db.m5.large",
    "db.m5.xlarge",
    "db.m5.2xlarge",
    "db.c6gd.large",
    "db.c6gd.xlarge",
    "db.c6gd.2xlarge",
    "db.r8g.large",
    "db.r8g.xlarge",
    "db.r8g.2xlarge",
    "db.x1.16xlarge",
    "db.x1.32xlarge",
    "db.r5.large",
    "db.r5.xlarge",
    "db.r5.2xlarge",
    "db.r5.4xlarge",
]

RDS_ENGINES = ["mysql", "postgres", "oracle-ee", "sqlserver-ee"]

RDS_STORAGE_TYPES = ["gp2", "io1", "standard"]


def run_cli_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result


class BaseServiceState(object):
    def __init__(self, property_names: Optional[List[str]] = None):
        self._property_names = property_names
        self.list_of_services = {}

    def add(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def ids(self):
        return list(self.list_of_services.keys())

    @property
    def num_items(self):
        return len(self.list_of_services.keys())

    def num_items_with_prefix(self, prefix):
        num_services = 0
        for service in self.list_of_services.keys():
            if service.startswith(prefix):
                num_services += 1
        return num_services

    def get_key(self, key: str):
        assert self._property_names is not None, "This service has no property to query"
        assert key in self._property_names
        return key

    def filter(self, service_name: str, kwarg: str):
        key = self.get_key(kwarg)
        if service_name not in self.list_of_services:
            raise AssertionError("Service does not exist")
        instance = self.list_of_services[service_name]
        return instance[key]

    def item_with_kwarg_enabled(self, kwarg: str, service_id: str):
        if service_id in self.list_of_services:
            return self.filter(service_name=service_id, kwarg=kwarg) > 0
        else:
            return False

    def num_items_with_kwarg_enabled(self, kwarg: str, prefix: Optional[str] = None):
        num_services = 0
        for service in self.list_of_services.keys():
            if prefix is not None:
                if service.startswith(prefix):
                    num_services += self.item_with_kwarg_enabled(
                        kwarg=kwarg, service_id=service
                    )
            else:
                num_services += self.item_with_kwarg_enabled(
                    kwarg=kwarg, service_id=service
                )
        return num_services

    def item_with_property(
        self, property: str, desired_property: Union[str, bool], service_id: str
    ):
        if service_id not in self.list_of_services:
            return False
        return self.filter(service_id, property) == desired_property

    def num_items_with_property(
        self,
        property: str,
        desired_property: Union[str, int],
        prefix: Optional[str] = None,
    ):
        num_services = 0
        for service in self.list_of_services.keys():
            if prefix is not None:
                if service.startswith(prefix):
                    num_services += self.item_with_property(
                        property=property,
                        desired_property=desired_property,
                        service_id=service,
                    )
            else:
                num_services += self.item_with_property(
                    property=property,
                    desired_property=desired_property,
                    service_id=service,
                )
        return num_services


class S3ServiceState(BaseServiceState):
    def __init__(self):
        super().__init__(property_names=["account control list", "object ownership"])

    def add(self, bucket_name: str, acl: str, object_ownership: str):
        if bucket_name not in self.list_of_services:
            self.list_of_services[bucket_name] = {
                "account control list": acl,
                "object ownership": object_ownership,
            }
        else:
            raise AssertionError("Bucket already exists")


class RDSInstance(BaseServiceState):
    def __init__(self):
        super().__init__(
            property_names=[
                "public access",
                "storage",
                "instance type",
                "storage type",
                "DB Engine",
                "encryption enabled",
            ]
        )

    def add(
        self,
        instance_name: str,
        public_access: bool,
        storage: int,
        instance_type: str,
        storage_type: str,
        db_engine: str,
        encryption_enabled: bool,
    ):
        if instance_name not in self.list_of_services:
            self.list_of_services[instance_name] = {
                "public access": public_access,
                "storage": storage,
                "instance type": instance_type,
                "storage type": storage_type,
                "DB Engine": db_engine,
                "encryption enabled": encryption_enabled,
            }
        else:
            raise AssertionError("Instance already exists")

    def num_items_with_storage(self, storage_kwarg: str):
        greater_than, less_than = False, False
        if storage_kwarg.startswith("greater than"):
            greater_than = True
            desired_storage = int(storage_kwarg.split("greater than ")[1])
        elif storage_kwarg.startswith("less than"):
            less_than = True
            desired_storage = int(storage_kwarg.split("less than ")[1])
        elif storage_kwarg.startswith("equal to"):
            desired_storage = int(storage_kwarg.split("equal to ")[1])
        else:
            raise NotImplementedError
        num_instance = 0
        for instance in self.list_of_services.keys():
            storage = self.filter(service_name=instance, kwarg="storage")
            if greater_than:
                num_instance += storage > desired_storage
            elif less_than:
                num_instance += storage < desired_storage
            else:
                num_instance += storage == desired_storage
        return num_instance

    def item_with_storage(self, storage_kwarg: str, service_id: str):
        if service_id not in self.list_of_services:
            return False
        greater_than, less_than = False, False
        if storage_kwarg.startswith("greater than"):
            greater_than = True
            desired_storage = int(storage_kwarg.split("greater than ")[1])
        elif storage_kwarg.startswith("less than"):
            less_than = True
            desired_storage = int(storage_kwarg.split("less than ")[1])
        elif storage_kwarg.startswith("equal to"):
            desired_storage = int(storage_kwarg.split("equal to ")[1])
        else:
            raise NotImplementedError
        storage = self.filter(service_name=service_id, kwarg="storage")
        if greater_than:
            return storage > desired_storage
        elif less_than:
            return storage < desired_storage
        else:
            return storage == desired_storage


class IAMRole(BaseServiceState):
    def __init__(self):
        properties = ["user_name", "ec2 access", "s3 access", "rds access"]
        super().__init__(property_names=properties)

    def add(
        self,
        role_name: str,
        user_name: str,
        ec2_access: bool,
        s3_access: bool,
        rds_access: bool,
    ):
        if role_name not in self.list_of_services:
            self.list_of_services[role_name] = {
                "user_name": user_name,
                "ec2 access": ec2_access,
                "s3 access": s3_access,
                "rds access": rds_access,
            }
        else:
            raise AssertionError("Role already exists")


class EC2Instance(BaseServiceState):
    def __init__(self):
        super().__init__(
            property_names=[
                "ami_id",
                "assign public ip",
                "state",
                "instance type",
                "monitoring enabled",
            ]
        )

    def add(
        self,
        instance_id: str,
        ami_id: str,
        assign_public_ip: bool,
        state: str,
        instance_type: str,
        monitoring_enabled: bool,
    ):
        if instance_id not in self.list_of_services:
            self.list_of_services[instance_id] = {
                "ami_id": ami_id,
                "assign public ip": assign_public_ip,
                "state": state,
                "instance type": instance_type,
                "monitoring enabled": monitoring_enabled,
            }
        else:
            raise AssertionError("Role already exists")

    def has_public_ip(self, instance_id: str):
        role = self.list_of_services[instance_id]
        if role["assign public ip"]:
            # has public IP only if we make it publicly accessible and the instance is running
            return role["state"] == "running"
        else:
            return False

    def get_key(self, key: str):
        assert key in [
            "ami_id",
            "AMI ID",
            "assign public ip",
            "public IP",
            "state",
            "instance type",
            "monitoring enabled",
        ], f"Property {key} not found"
        if key in ["ami_id", "AMI ID"]:
            return "ami_id"
        else:
            return key

    def filter(self, service_name: str, kwarg: str):
        key = self.get_key(kwarg)
        if service_name not in self.list_of_services:
            raise AssertionError("EC2 instance does not exist")

        if key == "public IP":
            # only running instances that are publicly accessible get a public IP
            return self.has_public_ip(service_name)
        else:
            service = self.list_of_services[service_name]
            return service[key]


class HiddenEnvState(object):
    def __init__(self):
        self._state = {
            "s3": S3ServiceState(),
            "rds": RDSInstance(),
            "iam": IAMRole(),
            "ec2": EC2Instance(),
        }

    def add_to_state(self, service_type: str, **kwargs):
        if service_type not in self._state.keys():
            raise AssertionError(f"Service {service_type} does not exist")
        self._state[service_type].add(**kwargs)

    def get_service(self, service_type: str) -> BaseServiceState:
        if service_type not in self._state.keys():
            raise AssertionError(f"Service {service_type} does not exist")
        return copy.deepcopy(self._state[service_type])


@dataclass
class QueryCombination:
    query_start: str
    services: List[str]
    service_properties: Dict[str, Dict]
    # list_of_possible_query_params: List[Tuple[List, int]]  # this has to be sorted.


@dataclass
class Query:
    query_text: str
    query_type: str
    service: str
    param: str
    service_id: Optional[str] = None
    queried_property: Optional[str] = None


class MotoServer:
    max_attempts: int = 10

    def __init__(self, port: int = 5000, region: str = "us-east-1"):
        self.port = port
        self.set_mocked_aws_credentials()
        self.server = self._create_server(port)
        self.endpoint = f"http://localhost:{port}"
        self._region = region
        self.subnet_id, self.sg = None, None

    @staticmethod
    def set_mocked_aws_credentials():
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
        os.environ["AWS_SESSION_TOKEN"] = "testing"
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"

    @staticmethod
    def _create_server(port):
        return subprocess.Popen(
            f"exec moto_server --port {port}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def set_region(self, region: str):
        self._region = region

    # def create_security_group(self):
    #     # TODO: This still seems to fail -- certain security groups are not deleted based on reset?
    #     base_command = [
    #         "aws",
    #         "--endpoint-url",
    #         self.endpoint,
    #     ]
    #     command = copy.deepcopy(base_command)
    #     command.extend(
    #         [
    #             "ec2",
    #             "create-vpc",
    #             "--region",
    #             self._region,
    #             "--cidr-block",
    #             "10.0.0.0/16",
    #         ]
    #     )
    #     result = run_cli_command(command=command)
    #     try:
    #         vpc_id = json.loads(result.stdout)["Vpc"]["VpcId"]
    #     except json.decoder.JSONDecodeError:
    #         raise AssertionError("received error:", result.stderr)
    #     command = copy.deepcopy(base_command)
    #     command.extend(
    #         [
    #             "ec2",
    #             "create-subnet",
    #             "--region",
    #             self._region,
    #             "--vpc-id",
    #             vpc_id,
    #             "--cidr-block",
    #             "10.0.0.0/16",
    #             "--region",
    #             self._region,
    #         ]
    #     )
    #     result = run_cli_command(command=command)
    #     try:
    #         subnet_id = json.loads(result.stdout)["Subnet"]["SubnetId"]
    #     except json.decoder.JSONDecodeError:
    #         raise AssertionError("received error:", result.stderr)
    #
    #     command = copy.deepcopy(base_command)
    #     command.extend(
    #         [
    #             "ec2",
    #             "create-security-group",
    #             "--region",
    #             self._region,
    #             "--group-name",
    #             f"TestGroup-{self.port}",
    #             "--description",
    #             "dummy security group",
    #         ]
    #     )
    #     result = run_cli_command(command=command)
    #     try:
    #         sg = json.loads(result.stdout)["GroupId"]
    #     except json.decoder.JSONDecodeError:
    #         raise AssertionError("received error:", result.stderr)
    #     return vpc_id, subnet_id, sg

    def get_default_security_group(self):
        base_command = [
            "aws",
            "--endpoint-url",
            self.endpoint,
        ]
        command = copy.deepcopy(base_command)
        command.extend(
            [
                "ec2",
                "describe-security-groups",
                "--region",
                self._region,
            ]
        )
        result = run_cli_command(command=command)
        try:
            groups = json.loads(result.stdout)["SecurityGroups"]
            group_id = groups[0]["GroupId"]
            return group_id
        except:
            print(f"No security group in region {self._region}")
            return None

    def get_default_subnet_id(self):
        base_command = [
            "aws",
            "--endpoint-url",
            self.endpoint,
        ]
        command = copy.deepcopy(base_command)
        command.extend(
            [
                "ec2",
                "describe-subnets",
                "--region",
                self._region,
            ]
        )
        result = run_cli_command(command=command)
        try:
            nets = json.loads(result.stdout)["Subnets"]
            subnet_id = nets[0]["SubnetId"]
            return subnet_id
        except:
            print(f"No subnets in region {self._region}")
            return None

    def delete_security_group(self):
        base_command = [
            "aws",
            "--endpoint-url",
            self.endpoint,
        ]
        command = copy.deepcopy(base_command)
        command.extend(
            [
                "ec2",
                "delete-security-group",
                "--region",
                self._region,
                "--group-id",
                f"{self.sg}",
            ]
        )
        result = run_cli_command(command=command)
        return True

    def reset_state(self):
        # resets the state of the server
        try:
            x = requests.post(
                url=f"{self.endpoint}/moto-api/reset", data=json.dumps({})
            )
            return True
        except:
            return False

    def kill(self):
        # self.reset_state()
        self.server.kill()
        self.server.wait()

    def _restart_server(self):
        reset_server = self.reset_state()
        if not reset_server:
            print("Requesting server restart failed, killing the server and restarting")
            self.kill()
            self.server = self._create_server(self.port)

    def restart(self):
        attempts = 0
        while True:
            self._restart_server()
            sg = self.get_default_security_group()
            subnet_id = self.get_default_subnet_id()
            if sg is not None and subnet_id is not None:
                break
            attempts += 1
            if attempts == self.max_attempts:
                raise AssertionError(
                    f"Tried to find default "
                    f"security group and subnet {self.max_attempts} times with no success"
                )
        # if sg is None or subnet_id is None:
        # self.kill()
        # self.server = self._create_server(port=self.port)
        #    vpc_id, self.subnet_id, self.sg = self.create_security_group()
        # else:
        self.sg = sg
        self.subnet_id = subnet_id
        self.endpoint = f"http://localhost:{self.port}"


def sentence_template():
    template = {
        "list_and_count": {
            "templates": [
                "How many [SERVICE] do I have?",
                "What is the quantity of my [SERVICE]?",
                "What's my total number of [SERVICE]?",
                "Number of my [SERVICE]?",
                "How many [SERVICE] do I own?",
                "Total number of my [SERVICE]?",
            ],
            "name": {"s3": "", "rds": "", "iam": "", "ec2": ""},
        },
        "list_and_count_by_prefix": {
            "templates": [
                "How many [SERVICE] have [NAME] starting with [VAL]?",
                "What is the quantity of [SERVICE] with [NAME] starting with [VAL]?",
                "What's my total of [SERVICE] with [NAME] starting with [VAL]?",
                "Number of my [SERVICE] that have [NAME] starting with [VAL]?",
                "How many [SERVICE] with [NAME] starting with [VAL] do I own?",
                "Total number of my [SERVICE] with [NAME] starting with [VAL]?",
            ],
            "name": {"s3": "bucket name", "rds": "db instance IDs", "iam": "role name"},
        },
        "list_and_count_by_kwarg_enabled": {
            "templates": [
                "How many [SERVICE] have [NAME] [VAL]?",
                "What is the quantity of [SERVICE] with [NAME] [VAL]?",
                "What's my total of [SERVICE] with [NAME] [VAL]?",
                "Number of my [SERVICE] that have [NAME] [VAL]?",
                "How many [SERVICE] with [NAME] [VAL] do I own?",
                "Total number of my [SERVICE] with [NAME] [VAL]?",
            ],
            "name": {"rds": "", "iam": "inline policy with", "ec2": ""},
        },
        "list_and_count_by_property": {
            "templates": [
                "How many [SERVICE] have [NAME] [VAL]?",
                "What is the quantity of [SERVICE] with [NAME] [VAL]?",
                "What's my total of [SERVICE] with [NAME] [VAL]?",
                "Number of my [SERVICE] that have [NAME] [VAL]?",
                "How many [SERVICE] with [NAME] [VAL] do I own?",
                "Total number of my [SERVICE] with [NAME] [VAL]?",
            ],
            "name": {
                "s3": {
                    "account control list": "account control list",
                    "object ownership": "object ownership",
                },
                "rds": {
                    "instance type": "instance type",
                    "DB Engine": "DB Engine",
                    "storage type": "storage type",
                },
                "ec2": {
                    "ami_id": "AMI ID",
                    "state": "state",
                    "instance type": "instance type",
                },
            },
        },
        "list_and_count_by_storage": {
            "templates": [
                "How many [SERVICE] have [NAME] [VAL]?",
                "What is the quantity of [SERVICE] with [NAME] [VAL]?",
                "What's my total of [SERVICE] with [NAME] [VAL]?",
                "Number of my [SERVICE] that have [NAME] [VAL]?",
                "How many [SERVICE] with [NAME] [VAL] do I own?",
                "Total number of my [SERVICE] with [NAME] [VAL]?",
            ],
            "name": {"rds": "storage"},
        },
        "check_if_any_by_prefix": {
            "templates": [
                "Do any [SERVICE] have [NAME] starting with [VAL]?",
                "Is there any [SERVICE] with [NAME] starting with [VAL]?",
                "Do I have any [SERVICE] with [NAME] [VAL]?",
                "Do I have at least one [SERVICE] that have [NAME] starting with [VAL]?",
                "Are there any [SERVICE] with [NAME] starting with [VAL]?",
                "Do I have any [SERVICE] with [NAME] starting with [VAL]?",
            ],
            "name": {"s3": "bucket name", "rds": "db instance IDs", "iam": "role name"},
        },
        "check_if_any_by_kwarg_enabled": {
            "templates": [
                "Do any [SERVICE] have [NAME] [VAL]?",
                "Do I have any [SERVICE] with [NAME] [VAL]?",
                "Are there any [SERVICE] with [NAME] [VAL]?",
                "Do I have at least one [SERVICE] that have [NAME] [VAL]?",
                "Is there any [SERVICE] with [NAME] [VAL]?",
                "Do I have at least one [SERVICE] with [NAME] [VAL]?",
            ],
            "name": {"rds": "", "iam": "inline policy with", "ec2": ""},
        },
        "check_if_any_by_property": {
            "templates": [
                "Do any [SERVICE] have [NAME] [VAL]?",
                "Do I have any [SERVICE] with [NAME] [VAL]?",
                "Are there any [SERVICE] with [NAME] [VAL]?",
                "Do I have at least one [SERVICE] that have [NAME] [VAL]?",
                "Is there any [SERVICE] with [NAME] [VAL] do I own?",
                "Do I have at least one [SERVICE] with [NAME] [VAL]?",
            ],
            "name": {
                "s3": {
                    "account control list": "account control list",
                    "object ownership": "object ownership",
                },
                "rds": {
                    "instance type": "instance type",
                    "DB Engine": "DB Engine",
                    "storage type": "storage type",
                },
                "ec2": {
                    "ami_id": "AMI ID",
                    "state": "state",
                    "instance type": "instance type",
                },
            },
        },
        "check_if_any_by_storage": {
            "templates": [
                "Do any [SERVICE] have [NAME] [VAL]?",
                "Do I have any [SERVICE] with [NAME] [VAL]?",
                "Are there any [SERVICE] with [NAME] [VAL]?",
                "Do I have at least one [SERVICE] that have [NAME] [VAL]?",
                "Is there any [SERVICE] with [NAME] [VAL] do I own?",
                "Do I have at least one [SERVICE] with [NAME] [VAL]?",
            ],
            "name": {"rds": "storage"},
        },
        "check_specific_if_kwarg_enabled": {
            "templates": [
                "Does the [SERVICE] [SERVICE_ID] have [NAME] [VAL]?",
                "Is there [NAME] [VAL] for [SERVICE] [SERVICE_ID]?",
                "[SERVICE] [SERVICE_ID] have [NAME] [VAL]?",
                "[NAME] [VAL] for [SERVICE] [SERVICE_ID]?",
            ],
            "name": {"rds": "", "iam": "inline policy with", "ec2": ""},
        },
        "check_specific_if_property": {
            "templates": [
                "Does the [SERVICE] [SERVICE_ID] have [NAME] [VAL]?",
                "Is there [NAME] [VAL] for [SERVICE] [SERVICE_ID]?",
                "[SERVICE] [SERVICE_ID] have [NAME] [VAL]?",
                "[NAME] [VAL] for [SERVICE] [SERVICE_ID]?",
            ],
            "name": {
                "s3": {
                    "account control list": "account control list",
                    "object ownership": "object ownership",
                },
                "rds": {
                    "instance type": "instance type",
                    "DB Engine": "DB Engine",
                    "storage type": "storage type",
                },
                "ec2": {
                    "ami_id": "AMI ID",
                    "state": "state",
                    "instance type": "instance type",
                },
            },
        },
        "check_specific_if_storage": {
            "templates": [
                "Does the [SERVICE] [SERVICE_ID] have [NAME] [VAL]?",
                "Is there [NAME] [VAL] for [SERVICE] [SERVICE_ID]?",
                "[SERVICE] [SERVICE_ID] have [NAME] [VAL]?",
                "[NAME] [VAL] for [SERVICE] [SERVICE_ID]?",
            ],
            "name": {"rds": "storage"},
        },
    }
    return template


def query_combo_list(query_cls: str = "easy"):
    # TODO: Check possible set of queries
    storage_queries = []
    assert query_cls in ["easy", "medium", "hard"]
    for storage in STORAGE_LIMITS:
        storage_queries.append(f"less than {storage}")
        storage_queries.append(f"greater than {storage}")
        storage_queries.append(f"equal to {storage}")
    if query_cls == "easy" or query_cls == "medium":
        query_combo = [
            QueryCombination(
                query_start="How many",
                services=["s3", "rds", "iam", "ec2"],
                service_properties={
                    "s3": {
                        "service_name": "S3 Buckets",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have bucket name starting with",
                                PREFIX,
                                "",
                                "list_and_count_by_prefix",
                            ),
                        ],
                    },
                    "rds": {
                        "service_name": "RDS instances",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have db instance IDs starting with",
                                PREFIX,
                                "",
                                "list_and_count_by_prefix",
                            ),
                            (
                                "have",
                                ["public access"],
                                "",
                                "list_and_count_by_kwarg_enabled",
                            ),
                            (
                                "have storage",
                                storage_queries,
                                "",
                                "list_and_count_by_storage",
                            ),
                        ],
                    },
                    "iam": {
                        "service_name": "IAM Roles",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have role name start with",
                                PREFIX,
                                "",
                                "list_and_count_by_prefix",
                            ),
                            (
                                "have inline policy with",
                                ["ec2 access", "s3 access", "rds access"],
                                "",
                                "list_and_count_by_kwarg_enabled",
                            ),
                        ],
                    },
                    "ec2": {
                        "service_name": "EC2 Instances",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have a",
                                ["public IP"],
                                "",
                                "list_and_count_by_kwarg_enabled",
                            ),
                            (
                                "are in state:",
                                EC2_STATES,
                                "state",
                                "list_and_count_by_property",
                            ),
                            (
                                "have AMI ID:",
                                AMI_IDS,
                                "ami_id",
                                "list_and_count_by_property",
                            ),
                        ],
                    },
                },
            ),
            QueryCombination(
                query_start="Do any",
                services=["s3", "rds", "iam", "ec2"],
                service_properties={
                    "s3": {
                        "service_name": "S3 Buckets",
                        "service_options": [
                            (
                                "have bucket name starting with",
                                PREFIX,
                                "",
                                "check_if_any_by_prefix",
                            ),
                        ],
                    },
                    "rds": {
                        "service_name": "RDS instances",
                        "service_options": [
                            (
                                "have db instance IDs starting with",
                                PREFIX,
                                "",
                                "check_if_any_by_prefix",
                            ),
                            (
                                "have",
                                ["public access"],
                                "",
                                "check_if_any_by_kwarg_enabled",
                            ),
                            (
                                "have storage",
                                storage_queries,
                                "",
                                "check_if_any_by_storage",
                            ),
                        ],
                    },
                    "iam": {
                        "service_name": "IAM Roles",
                        "service_options": [
                            (
                                "have role name starting with",
                                PREFIX,
                                "",
                                "check_if_any_by_prefix",
                            ),
                            (
                                "have inline policy with",
                                ["ec2 access", "s3 access", "rds access"],
                                "",
                                "check_if_any_by_kwarg_enabled",
                            ),
                        ],
                    },
                    "ec2": {
                        "service_name": "EC2 Instances",
                        "service_options": [
                            (
                                "have a",
                                ["public IP"],
                                "",
                                "check_if_any_by_kwarg_enabled",
                            ),
                            (
                                "have the state:",
                                EC2_STATES,
                                "state",
                                "check_if_any_by_property",
                            ),
                            (
                                "have AMI ID:",
                                AMI_IDS,
                                "ami_id",
                                "check_if_any_by_property",
                            ),
                        ],
                    },
                },
            ),
            QueryCombination(
                query_start="Does the",
                services=["rds", "iam", "ec2"],
                service_properties={
                    "rds": {
                        "service_name": "RDS instance",
                        "service_options": [
                            (
                                "have",
                                ["public access"],
                                "",
                                "check_specific_if_kwarg_enabled",
                            ),
                            (
                                "have storage",
                                storage_queries,
                                "",
                                "check_specific_if_storage",
                            ),
                        ],
                    },
                    "iam": {
                        "service_name": "IAM Role",
                        "service_options": [
                            (
                                "have an inline policy with",
                                ["ec2 access", "s3 access", "rds access"],
                                "",
                                "check_specific_if_kwarg_enabled",
                            ),
                        ],
                    },
                    "ec2": {
                        "service_name": "EC2 Instance",
                        "service_options": [
                            (
                                "have a",
                                ["public IP"],
                                "",
                                "check_specific_if_kwarg_enabled",
                            ),
                            (
                                "have the state:",
                                EC2_STATES,
                                "state",
                                "check_specific_if_property",
                            ),
                            (
                                "have the AMI ID:",
                                AMI_IDS,
                                "ami_id",
                                "check_specific_if_property",
                            ),
                        ],
                    },
                },
            ),
        ]
    else:
        query_combo = [
            QueryCombination(
                query_start="How many",
                services=["s3", "rds", "iam", "ec2"],
                service_properties={
                    "s3": {
                        "service_name": "S3 Buckets",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have bucket name starting with",
                                PREFIX,
                                "",
                                "list_and_count_by_prefix",
                            ),
                            (
                                "have account control list:",
                                S3_ACL_OPTIONS,
                                "account control list",
                                "list_and_count_by_property",
                            ),
                            (
                                "have object ownership:",
                                S3_OBJECT_OWNERSHIP_OPTIONS,
                                "object ownership",
                                "list_and_count_by_property",
                            ),
                        ],
                    },
                    "rds": {
                        "service_name": "RDS instances",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have db instance IDs starting with",
                                PREFIX,
                                "",
                                "list_and_count_by_prefix",
                            ),
                            (
                                "have",
                                ["public access", "encryption enabled"],
                                "",
                                "list_and_count_by_kwarg_enabled",
                            ),
                            (
                                "have instance type:",
                                RDS_INSTANCE_TYPES,
                                "instance type",
                                "list_and_count_by_property",
                            ),
                            (
                                "have DB Engine:",
                                RDS_ENGINES,
                                "DB Engine",
                                "list_and_count_by_property",
                            ),
                            (
                                "have storage type:",
                                RDS_STORAGE_TYPES,
                                "storage type",
                                "list_and_count_by_property",
                            ),
                            (
                                "have storage",
                                storage_queries,
                                "",
                                "list_and_count_by_storage",
                            ),
                        ],
                    },
                    "iam": {
                        "service_name": "IAM Roles",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have role name start with",
                                PREFIX,
                                "",
                                "list_and_count_by_prefix",
                            ),
                            (
                                "have inline policy with",
                                ["ec2 access", "s3 access", "rds access"],
                                "",
                                "list_and_count_by_kwarg_enabled",
                            ),
                        ],
                    },
                    "ec2": {
                        "service_name": "EC2 Instances",
                        "service_options": [
                            ("do I have", [""], [""], "list_and_count"),
                            (
                                "have",
                                ["public IP", "monitoring enabled"],
                                "",
                                "list_and_count_by_kwarg_enabled",
                            ),
                            (
                                "are in state:",
                                EC2_STATES,
                                "state",
                                "list_and_count_by_property",
                            ),
                            (
                                "have AMI ID:",
                                AMI_IDS,
                                "ami_id",
                                "list_and_count_by_property",
                            ),
                            (
                                "have instance type:",
                                EC2_INSTANCE_TYPES,
                                "instance type",
                                "list_and_count_by_property",
                            ),
                        ],
                    },
                },
            ),
            QueryCombination(
                query_start="Do any",
                services=["s3", "rds", "iam", "ec2"],
                service_properties={
                    "s3": {
                        "service_name": "S3 Buckets",
                        "service_options": [
                            (
                                "have bucket name starting with",
                                PREFIX,
                                "",
                                "check_if_any_by_prefix",
                            ),
                            (
                                "have account control list:",
                                S3_ACL_OPTIONS,
                                "account control list",
                                "check_if_any_by_property",
                            ),
                            (
                                "have object ownership:",
                                S3_OBJECT_OWNERSHIP_OPTIONS,
                                "object ownership",
                                "check_if_any_by_property",
                            ),
                        ],
                    },
                    "rds": {
                        "service_name": "RDS instances",
                        "service_options": [
                            (
                                "have db instance IDs starting with",
                                PREFIX,
                                "",
                                "check_if_any_by_prefix",
                            ),
                            (
                                "have",
                                ["public access", "encryption enabled"],
                                "",
                                "check_if_any_by_kwarg_enabled",
                            ),
                            (
                                "have instance type:",
                                RDS_INSTANCE_TYPES,
                                "instance type",
                                "check_if_any_by_property",
                            ),
                            (
                                "have DB Engine:",
                                RDS_ENGINES,
                                "DB Engine",
                                "check_if_any_by_property",
                            ),
                            (
                                "have storage type:",
                                RDS_STORAGE_TYPES,
                                "storage type",
                                "check_if_any_by_property",
                            ),
                            (
                                "have storage",
                                storage_queries,
                                "",
                                "check_if_any_by_storage",
                            ),
                        ],
                    },
                    "iam": {
                        "service_name": "IAM Roles",
                        "service_options": [
                            (
                                "have role name starting with",
                                PREFIX,
                                "",
                                "check_if_any_by_prefix",
                            ),
                            (
                                "have inline policy with",
                                ["ec2 access", "s3 access", "rds access"],
                                "",
                                "check_if_any_by_kwarg_enabled",
                            ),
                        ],
                    },
                    "ec2": {
                        "service_name": "EC2 Instances",
                        "service_options": [
                            (
                                "have",
                                ["public IP", "monitoring enabled"],
                                "",
                                "check_if_any_by_kwarg_enabled",
                            ),
                            (
                                "have the state:",
                                EC2_STATES,
                                "state",
                                "check_if_any_by_property",
                            ),
                            (
                                "have instance type:",
                                EC2_INSTANCE_TYPES,
                                "instance type",
                                "check_if_any_by_property",
                            ),
                            (
                                "have AMI ID:",
                                AMI_IDS,
                                "ami_id",
                                "check_if_any_by_property",
                            ),
                        ],
                    },
                },
            ),
            QueryCombination(
                query_start="Does the",
                services=["s3", "rds", "iam", "ec2"],
                service_properties={
                    "s3": {
                        "service_name": "S3 Buckets",
                        "service_options": [
                            (
                                "have account control list:",
                                S3_ACL_OPTIONS,
                                "account control list",
                                "check_specific_if_property",
                            ),
                            (
                                "have object ownership:",
                                S3_OBJECT_OWNERSHIP_OPTIONS,
                                "object ownership",
                                "check_specific_if_property",
                            ),
                        ],
                    },
                    "rds": {
                        "service_name": "RDS instance",
                        "service_options": [
                            (
                                "have",
                                ["public access", "encryption enabled"],
                                "",
                                "check_specific_if_kwarg_enabled",
                            ),
                            (
                                "have instance type:",
                                RDS_INSTANCE_TYPES,
                                "instance type",
                                "check_specific_if_property",
                            ),
                            (
                                "have DB Engine:",
                                RDS_ENGINES,
                                "DB Engine",
                                "check_specific_if_property",
                            ),
                            (
                                "have storage type:",
                                RDS_STORAGE_TYPES,
                                "storage type",
                                "check_specific_if_property",
                            ),
                            (
                                "have storage",
                                storage_queries,
                                "",
                                "check_specific_if_storage",
                            ),
                        ],
                    },
                    "iam": {
                        "service_name": "IAM Role",
                        "service_options": [
                            (
                                "have an inline policy with",
                                ["ec2 access", "s3 access", "rds access"],
                                "",
                                "check_specific_if_kwarg_enabled",
                            ),
                        ],
                    },
                    "ec2": {
                        "service_name": "EC2 Instance",
                        "service_options": [
                            (
                                "have",
                                ["public IP", "monitoring enabled"],
                                "",
                                "check_specific_if_kwarg_enabled",
                            ),
                            (
                                "have instance type:",
                                EC2_INSTANCE_TYPES,
                                "instance type",
                                "check_specific_if_property",
                            ),
                            (
                                "have the state:",
                                EC2_STATES,
                                "state",
                                "check_specific_if_property",
                            ),
                            (
                                "have the AMI ID:",
                                AMI_IDS,
                                "ami_id",
                                "check_specific_if_property",
                            ),
                        ],
                    },
                },
            ),
        ]

    return query_combo
