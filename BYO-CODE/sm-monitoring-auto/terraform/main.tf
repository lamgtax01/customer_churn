# S3 Buckets - Where Lambda code is stored
resource "aws_s3_bucket" "lambda_code_bucket" {
  bucket = "s3-${var.env}-use1-mrm240002-lambda-code"

  tags = {
    Name        = "s3-${var.env}-use1-mrm240002-lambda-code"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lambda_code_encryption" {
  bucket = aws_s3_bucket.lambda_code_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}


#  IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "lam-${var.env}-use1-sagemaker-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_policy" "lambda_policy" {
  name = "LambdaSageMakerPolicy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:us-east-1:047922237497:log-group:/aws/lambda/lam-${var.env}-use1-mrm240002-datadrift:*",
          "arn:aws:logs:us-east-1:047922237497:log-group:/aws/lambda/lam-${var.env}-use1-mrm240002-modeldrift:*"
        ]
      },
      {
        Effect = "Allow"
        Action = "sagemaker:StartPipelineExecution"
        Resource = [
          "arn:aws:sagemaker:us-east-1:047922237497:pipeline/DataDriftPipeline",
          "arn:aws:sagemaker:us-east-1:047922237497:pipeline/ModelDriftPipeline"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_attach" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}


#  Lambda Function (Data Drift)
data "archive_file" "lambda_datadrift_zip" {
  type        = "zip"
  source_file = "./lambda/lambda_datadrift.py"
  output_path = "./lambda/lambda_datadrift.zip"
}

resource "aws_lambda_function" "data_drift_lambda" {
  function_name    = "lam-${var.env}-use1-mrm240002-datadrift"
  role             = aws_iam_role.lambda_role.arn
  handler          = "lambda_datadrift.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.lambda_datadrift_zip.output_path
  source_code_hash = data.archive_file.lambda_datadrift_zip.output_base64sha256

  environment {
    variables = {
      DATA_DRIFT_PIPELINE = "DataDriftPipeline"
    }
  }
}

# Lambda Function (Model Drift)
data "archive_file" "lambda_modeldrift_zip" {
  type        = "zip"
  source_file = "./lambda/lambda_modeldrift.py"
  output_path = "./lambda/lambda_modeldrift.zip"
}

# Deploy the zipped Lambda function
resource "aws_lambda_function" "model_drift_lambda" {
  function_name    = "lam-${var.env}-use1-mrm240002-modeldrift"
  role             = aws_iam_role.lambda_role.arn
  handler          = "lambda_modeldrift.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.lambda_modeldrift_zip.output_path
  source_code_hash = data.archive_file.lambda_modeldrift_zip.output_base64sha256

  environment {
    variables = {
      MODEL_DRIFT_PIPELINE = "ModelDriftPipeline"
    }
  }
}


# S3 Event Notifications - data Drift
resource "aws_s3_bucket_notification" "s3_event_inf" {
  bucket = var.data_bucket

  lambda_function {
    lambda_function_arn = aws_lambda_function.data_drift_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".csv"
    filter_prefix       = "${var.folder_mrm2400002_INF}/"
  }
}

resource "aws_lambda_permission" "s3_invoke_data_drift" {
  statement_id  = "AllowS3InvokeDataDrift"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.data_drift_lambda.arn
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${var.data_bucket}"
}


# S3 Event Notifications - Model Drift
resource "aws_s3_bucket_notification" "s3_event_grt" {
  bucket = var.data_bucket

  lambda_function {
    lambda_function_arn = aws_lambda_function.model_drift_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".csv"
    filter_prefix       = "${var.folder_mrm2400002_GT}/"
  }
}

resource "aws_lambda_permission" "s3_invoke_model_drift" {
  statement_id  = "AllowS3InvokeModelDrift"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.model_drift_lambda.arn
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${var.data_bucket}"
}

