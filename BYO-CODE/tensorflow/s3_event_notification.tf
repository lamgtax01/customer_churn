# Loop through all S3 buckets specified in the manifest
resource "aws_s3_bucket_notification" "s3_event_notifications" {
  for_each = { for idx, bucket in var.s3_buckets : bucket.name => bucket }

  bucket = each.value.name

  lambda_function {
    lambda_function_arn = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${each.value.lambda_function}"
    events              = [each.value.event_type]
  }
}

# Ensure Lambda permission is added for each S3 bucket
resource "aws_lambda_permission" "s3_invoke_lambda" {
  for_each = { for idx, bucket in var.s3_buckets : bucket.name => bucket }

  statement_id  = "AllowS3Invoke-${each.value.name}"
  action        = "lambda:InvokeFunction"
  function_name = each.value.lambda_function
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${each.value.name}"
}

data "aws_caller_identity" "current" {}


## Var
variable "aws_region" {
  default = "us-east-1"
}

variable "s3_buckets" {
  type = list(object({
    name            = string
    lambda_function = string
    event_type      = string
  }))
}

## yml
s3_buckets:
  - name: "bucket1"
    lambda_function: "lambda-a"
    event_type: "s3:ObjectCreated:*"

  - name: "bucket2"
    lambda_function: "lambda-b"
    event_type: "s3:ObjectRemoved:*"
