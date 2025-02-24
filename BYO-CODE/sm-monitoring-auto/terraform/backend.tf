terraform {
  backend "s3" {
    bucket         = "s3-dev-use1-mrm240002-remote-state"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "dynamo-dev-use1-mrm240002-remote-state"
    encrypt        = true
  }
}
