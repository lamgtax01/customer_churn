#!/bin/bash

ENV=$1  # Pass 'dev', 'qa', 'test', or 'prod' as the first argument

if [[ -z "$ENV" ]]; then
  echo "Usage: ./init.sh <env>"
  exit 1
fi

terraform init \
  -backend-config="bucket=s3-${ENV}-use1-mrm240002-remote-state" \
  -backend-config="dynamodb_table=dynamo-${ENV}-use1-mrm240002-remote-state"
