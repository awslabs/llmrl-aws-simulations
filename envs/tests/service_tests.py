from envs.moto_cli_env import AWSEnv
from envs.account_utils import run_cli_command
import json
import copy
from typing import Dict

NUM_RESET_TO_CHECK = 4


def check_acl(grants: Dict, owner: Dict) -> str:
    owner_id = owner["ID"]
    # Check for private ACL
    if (
        len(grants) == 1
        and grants[0]["Grantee"]["Type"] == "CanonicalUser"
        and grants[0]["Permission"] == "FULL_CONTROL"
        and grants[0]["Grantee"]["ID"] == owner_id
    ):
        return "private"

    # Check for public-read ACL
    elif (
        len(grants) == 2
        and any(
            g["Grantee"]["Type"] == "CanonicalUser"
            and g["Permission"] == "FULL_CONTROL"
            and g["Grantee"]["ID"] == owner_id
            for g in grants
        )
        and any(
            g["Grantee"]["Type"] == "Group"
            and g["Grantee"]["URI"] == "http://acs.amazonaws.com/groups/global/AllUsers"
            and g["Permission"] == "READ"
            for g in grants
        )
    ):
        return "public-read"

    # Check for public-read-write ACL
    elif (
        len(grants) == 3
        and any(
            g["Grantee"]["Type"] == "CanonicalUser"
            and g["Permission"] == "FULL_CONTROL"
            and g["Grantee"]["ID"] == owner_id
            for g in grants
        )
        and any(
            g["Grantee"]["Type"] == "Group"
            and g["Grantee"]["URI"] == "http://acs.amazonaws.com/groups/global/AllUsers"
            and g["Permission"] == "READ"
            for g in grants
        )
        and any(
            g["Grantee"]["Type"] == "Group"
            and g["Grantee"]["URI"] == "http://acs.amazonaws.com/groups/global/AllUsers"
            and g["Permission"] == "WRITE"
            for g in grants
        )
    ):
        return "public-read-write"

    # Check for authenticated-read ACL
    elif (
        len(grants) == 2
        and any(
            g["Grantee"]["Type"] == "CanonicalUser"
            and g["Permission"] == "FULL_CONTROL"
            and g["Grantee"]["ID"] == owner_id
            for g in grants
        )
        and any(
            g["Grantee"]["Type"] == "Group"
            and g["Grantee"]["URI"]
            == "http://acs.amazonaws.com/groups/global/AuthenticatedUsers"
            and g["Permission"] == "READ"
            for g in grants
        )
    ):
        return "authenticated-read"

    else:
        return "custom"


def test_s3():
    env = AWSEnv(
        max_num_buckets=10,
        max_num_rds=5,
        max_num_roles=5,
        max_num_ec2s=5,
    )
    num_resets = NUM_RESET_TO_CHECK
    for i in range(num_resets):
        print(f"Testing with S3 at iteration: {i}")
        env.reset()
        execute_command = copy.deepcopy(env.base_command)
        execute_command.extend(["--region", env.region])
        execute_command.extend(["s3api", "list-buckets"])
        out = run_cli_command(execute_command)
        result = json.loads(out.stdout)
        state = env.hidden_state.get_service(service_type="s3")
        num_buckets = 0
        for bucket in result["Buckets"]:
            bucket_name = bucket["Name"]
            assert (
                bucket_name in state.list_of_services
            ), "Bucket {} does not exist".format(bucket_name)
            execute_command = copy.deepcopy(env.base_command)
            execute_command.extend(["--region", env.region])
            execute_command.extend(
                [
                    "s3api",
                    "get-bucket-acl",
                    "--bucket",
                    bucket_name,
                    "--output",
                    "json",
                ]
            )
            out = run_cli_command(execute_command)
            acl_result = json.loads(out.stdout)
            acl = check_acl(acl_result["Grants"], acl_result["Owner"])
            assert (
                state.list_of_services[bucket_name]["account control list"] == acl
            ), "ACL type of the S3 bucket {bucket_name} do not match.".format(
                bucket_name=bucket_name
            )
            execute_command = copy.deepcopy(env.base_command)
            execute_command.extend(["--region", env.region])
            execute_command.extend(
                [
                    "s3api",
                    "get-bucket-ownership-controls",
                    "--bucket",
                    bucket_name,
                    "--output",
                    "json",
                ]
            )
            out = run_cli_command(execute_command)
            owner_result = json.loads(out.stdout)
            owner_result = owner_result["OwnershipControls"]["Rules"][0][
                "ObjectOwnership"
            ]
            assert (
                state.list_of_services[bucket_name]["object ownership"] == owner_result
            ), "Object ownership of the S3 bucket {bucket_name} do not match.".format(
                bucket_name=bucket_name
            )
            num_buckets += 1
        assert num_buckets == len(state.list_of_services), (
            "Number of "
            "buckets on Moto {} does "
            "not match number of buckets in "
            "env state {}"
        ).format(num_buckets, len(state.list_of_services))
        env.close()


def test_rds():
    env = AWSEnv(
        max_num_buckets=10,
        max_num_rds=5,
        max_num_roles=5,
        max_num_ec2s=5,
    )
    num_resets = NUM_RESET_TO_CHECK
    for i in range(num_resets):
        print(f"Testing with RDS at iteration: {i}")
        env.reset()
        execute_command = copy.deepcopy(env.base_command)
        execute_command.extend(["--region", env.region])
        execute_command.extend(["rds", "describe-db-instances"])
        out = run_cli_command(execute_command)
        result = json.loads(out.stdout)
        state = env.hidden_state.get_service(service_type="rds")
        num_instances = 0
        for instance in result["DBInstances"]:
            instance_id = instance["DBInstanceIdentifier"]
            assert (
                instance_id in state.list_of_services
            ), "Instance {} does not exist".format(instance_id)
            assert (
                state.list_of_services[instance_id]["storage"]
                == instance["AllocatedStorage"]
            ), ("Allocated Storage " "for {} does not " "match").format(instance_id)
            assert (
                state.list_of_services[instance_id]["public access"]
                == instance["PubliclyAccessible"]
            ), ("Public Access " "for {} does not " "match").format(instance_id)

            assert (
                state.list_of_services[instance_id]["instance type"]
                == instance["DBInstanceClass"]
            ), ("Instance type " "for {} does not " "match").format(instance_id)

            assert (
                    state.list_of_services[instance_id]["DB Engine"]
                    == instance["Engine"]
            ), ("DB Engine " "for {} does not " "match").format(instance_id)

            assert (
                    state.list_of_services[instance_id]["storage type"]
                    == instance["StorageType"]
            ), ("Storage type " "for {} does not " "match").format(instance_id)

            assert (
                    state.list_of_services[instance_id]["encryption enabled"]
                    == instance["StorageEncrypted"]
            ), ("Storage encryption " "for {} does not " "match").format(instance_id)
            num_instances += 1
        assert num_instances == len(state.list_of_services), (
            "Number of "
            "instances on Moto {} does "
            "not match number of instances in "
            "env state {}"
        ).format(num_instances, len(state.list_of_services))
        env.close()


def test_role_names():
    env = AWSEnv(
        max_num_buckets=10,
        max_num_rds=5,
        max_num_roles=5,
        max_num_ec2s=5,
        use_default_policy_names=True,
    )
    num_resets = NUM_RESET_TO_CHECK
    for i in range(num_resets):
        print(f"Testing with IAM at iteration: {i}")
        env.reset()
        execute_command = copy.deepcopy(env.base_command)
        execute_command.extend(["--region", env.region])
        execute_command.extend(["iam", "list-roles"])
        out = run_cli_command(execute_command)
        result = json.loads(out.stdout)
        state = env.hidden_state.get_service(service_type="iam")
        num_roles = 0
        for role in result["Roles"]:
            role_name = role["RoleName"]
            assert (
                role_name in state.list_of_services
            ), "Instance {} does not exist".format(role_name)
            num_roles += 1
            execute_command = copy.deepcopy(env.base_command)
            execute_command.extend(["--region", env.region])
            execute_command.extend(
                ["iam", "list-role-policies", "--role-name", role_name]
            )
            out = run_cli_command(execute_command)
            result = json.loads(out.stdout)["PolicyNames"]
            if state.list_of_services[role_name]["ec2 access"]:
                assert "EC2Access" in result, (
                    "EC2Access is saved in hidden state but "
                    "not found for role {role_name}"
                ).format(role_name=role_name)

            if state.list_of_services[role_name]["s3 access"]:
                assert "S3Access" in result, (
                    "S3Access is saved in hidden state but "
                    "not found for role {role_name}"
                ).format(role_name=role_name)

            if state.list_of_services[role_name]["rds access"]:
                assert "RDSAccess" in result, (
                    "RDSAccess is saved in hidden state but "
                    "not found for role {role_name}"
                ).format(role_name=role_name)

        assert num_roles == len(state.list_of_services), (
            "Number of "
            "instances on Moto {} does "
            "not match number of instances in "
            "env state {}"
        ).format(num_roles, len(state.list_of_services))
        env.close()


def test_ec2_ids():
    # TODO: Check public, resources and state.
    env = AWSEnv(
        max_num_buckets=10,
        max_num_rds=5,
        max_num_roles=5,
        max_num_ec2s=20,
    )
    num_resets = NUM_RESET_TO_CHECK
    for i in range(num_resets):
        print(f"Testing with EC2 at iteration: {i}")
        env.reset()
        execute_command = copy.deepcopy(env.base_command)
        query = (
            "Reservations[].Instances[].{"
            "InstanceId: InstanceId, State: State.Name,"
            "InstanceType: InstanceType,"
            "PublicIpAddress: PublicIpAddress, PrivateIpAddress: PrivateIpAddress,"
            "AmiId: ImageId, Monitoring: Monitoring.State"
            "}"
        )
        execute_command.extend(["--region", env.region])
        execute_command.extend(
            [
                "ec2",
                "describe-instances",
                "--query",
                query,
            ]
        )
        out = run_cli_command(execute_command)
        result = json.loads(out.stdout)
        state = env.hidden_state.get_service(service_type="ec2")
        num_instances = 0
        for instance in result:
            instance_id = instance["InstanceId"]
            assert (
                instance_id in state.list_of_services
            ), "Instance {} does not exist".format(instance_id)
            num_instances += 1
            assert (
                state.list_of_services[instance_id]["state"] == instance["State"]
            ), "States of the EC2 instance {instance_id} do not match.".format(
                instance_id=instance_id
            )
            assert (
                state.list_of_services[instance_id]["ami_id"] == instance["AmiId"]
            ), "AMI ID of the EC2 instance {instance_id} do not match.".format(
                instance_id=instance_id
            )
            assert (
                state.list_of_services[instance_id]["instance type"]
                == instance["InstanceType"]
            ), "Instance type of the EC2 instance {instance_id} do not match.".format(
                instance_id=instance_id
            )

            encryption = (
                "enabled"
                if state.list_of_services[instance_id]["monitoring enabled"]
                else "disabled"
            )
            assert (
                encryption == instance["Monitoring"]
            ), "Encryption status of the EC2 instance {instance_id} do not match.".format(
                instance_id=instance_id
            )
            if state.filter(instance_id, "public IP"):
                assert instance["PublicIpAddress"] is not None, (
                    "EC2 instance {instance_id} does not have public access "
                    "but it should.".format(instance_id=instance_id)
                )
            else:
                assert (
                    instance["PublicIpAddress"] is None
                ), "EC2 instance {instance_id} has public access but it shouldn't.".format(
                    instance_id=instance_id
                )

        assert num_instances == len(state.list_of_services), (
            "Number of "
            "instances on Moto {} does "
            "not match number of instances in "
            "env state {}"
        ).format(num_instances, len(state.list_of_services))
        env.close()


if __name__ == "__main__":
    test_s3()
    test_rds()
    test_role_names()
    test_ec2_ids()
