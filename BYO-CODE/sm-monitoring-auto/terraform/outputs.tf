output "lambda_data_drift_arn" {
  description = "ARN of the Data Drift Lambda function"
  value       = aws_lambda_function.data_drift_lambda.arn
}

output "lambda_data_drift_invoke_arn" {
  description = "Invoke ARN for Data Drift Lambda"
  value       = aws_lambda_function.data_drift_lambda.invoke_arn
}

output "lambda_data_drift_log_group" {
  description = "CloudWatch Log Group for Data Drift Lambda"
  value       = "/aws/lambda/${aws_lambda_function.data_drift_lambda.function_name}"
}

output "lambda_model_drift_arn" {
  description = "ARN of the Model Drift Lambda function"
  value       = aws_lambda_function.model_drift_lambda.arn
}

output "lambda_model_drift_invoke_arn" {
  description = "Invoke ARN for Model Drift Lambda"
  value       = aws_lambda_function.model_drift_lambda.invoke_arn
}

output "lambda_model_drift_log_group" {
  description = "CloudWatch Log Group for Model Drift Lambda"
  value       = "/aws/lambda/${aws_lambda_function.model_drift_lambda.function_name}"
}
