# Description of the MOTO CLI env

## Goal/Objective of the environment

We implement an environment to simulate the AWS CLI API.
This simulation is used as a tool to collect data for training LLM agents.
Effectively, given a user query regarding their AWS account, we prompt the LLM agent
to respond to the user's request by interacting with the AWS server via CLI commands.

## Modelling environment interactions

There are two crucial functions that we need in our environment.

1. Reset: This function is used to reset the account state and sample a new user query.
2. Step: This function takes as input the LLM's
   proposed action and if the action satisfies our specified syntax executes
   it as a CLI command in the AWS environment.
   We append the action and the response from the AWS side (output or error received from the server)
   to the state/context for the LLM. Conditioned on this new information, the LLM proposes the next action.
   The LLM agent is allowed to interact with the environment for at most 10 steps.
   Following which the environment is reset.

## High level setup for Moto

To execute CLI commands via Moto. We boot up a dummy server with a specific port ID.
Each server has a unique state and for each account we keep a unique port ID.
All requests associated to a specific account are routed to its server via the port ID.
To delete/reset the account state, we close the server.

## Types of AWS accounts modelled in the simulator

Each account is equipped with these services: S3, RDS, EC2 and IAM.
In the reset function, we do the following for each service.

1. S3: We sample a random number of buckets and for each bucket a random name starting with a specific prefix.
2. RDS: We sample a random number of RDS instances and for each instance a random name starting with a specific prefix.
   Furthermore, each instance is equipped with a randomly sampled storage limit, and we also
   randomly assign it to have public access.
3. IAM roles: We sample a random number of IAM roles and for each role a random name starting with a specific prefix.
   Furthermore, for each role, we sample a random username, and we also randomly assign it to have EC2, S3 and RDS
   access.
4. EC2: We sample a random AMI ID and assign public access, resource and an ec2 state (running, terminated, stopped)
   at random to each instance.

## Picking user queries

We restrict our set of queries to those which have numerical answers, e.g., 'How many RDS instance have prefix abc?'.
Each query is represented by a query text, query type, service and parameters associated with the query.
For example, the query: 'How many S3 buckets have prefix abc?'. Will be of query type 'list_and_count_by_prefix',
will have S3 as its service and the prefix 'abc' as the param.
To sample a query, we define a unique data structure which represents all possible query combinations.
Each query combination has a query start, e.g., 'How many', services (S3, RDS, IAM, EC2), and service properties,
which stores queries associated to the service.

## Finding the ground truth answer

To find the ground truth answer, we require 2 things.

1. A condensed representation of the AWS state. This stores all services associated with an account
   and all relevant attributes for the services that are relevant to answer the user query. For example, for S3 buckets,
   it would store the name of every bucket. For RDS instances, the name, storage and public access information.
   Note that this state is hidden from the LLM agent and it can only access this information by interacting with the AWS
   account.
   We only store this state to find the ground truth response for the reward function.
2. A function which maps the user query to the ground truth response stored in the state.


