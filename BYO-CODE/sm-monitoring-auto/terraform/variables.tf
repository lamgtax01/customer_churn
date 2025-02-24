variable "env" {
  description = "Environment name (dev, qa, test, prod)"
  type        = string
  default     = "dev"
}

variable "env_letter" {
  description = "Environment letter based on env"
  type        = map(string)
  default = {
    dev  = "d"
    qa   = "q"
    test = "t"
    prod = "p"
  }
}

variable "data_bucket" {
  description = "Pre-existing S3 bucket for notifications"
  type        = string
  default     = "aaa-dev-sm-monitoring-01"
}

variable "folder_mrm2400002_INF" {
  description = "S3 folder for Data Drift"
  type        = string
  default     = "mrm2400002_INF"
}

variable "folder_mrm2400002_GT" {
  description = "S3 folder for Model Drift"
  type        = string
  default     = "mrm2400002_GT"
}
