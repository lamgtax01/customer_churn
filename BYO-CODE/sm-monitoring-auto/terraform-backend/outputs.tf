output "s3_backend_bucket" {
  value = aws_s3_bucket.terraform_backend.bucket
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.terraform_lock.name
}
