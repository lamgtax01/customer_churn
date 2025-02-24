# 🔹 Create S3 Bucket for Remote State
resource "aws_s3_bucket" "terraform_backend" {
  bucket = "s3-${var.env}-use1-mrm240002-remote-state"

  lifecycle {
    prevent_destroy = true  # Prevent accidental deletion
  }

  tags = {
    Name        = "s3-${var.env}-use1-mrm240002-remote-state"
    Environment = var.env
  }
}

# 🔹 Enable Versioning (to track state changes)
resource "aws_s3_bucket_versioning" "versioning_enabled" {
  bucket = aws_s3_bucket.terraform_backend.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 🔹 Enable Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "s3_encryption" {
  bucket = aws_s3_bucket.terraform_backend.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 🔹 Create DynamoDB Table for State Locking
resource "aws_dynamodb_table" "terraform_lock" {
  name         = "dynamo-${var.env}-use1-mrm240002-remote-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name        = "dynamo-${var.env}-use1-mrm240002-remote-state"
    Environment = var.env
  }
}
