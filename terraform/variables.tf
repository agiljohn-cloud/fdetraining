variable "project_id" {
  description = "Google Cloud Project ID hosting the Meal Planning Concierge Agent."
  type        = string
  default     = "agiljohn-cert-oct2026"
}

variable "region" {
  description = "Primary Google Cloud region for Vertex AI Agent Engine and Cloud Run."
  type        = string
  default     = "us-east1"
}

variable "service_name" {
  description = "Name of the Cloud Run v2 service."
  type        = string
  default     = "meal-concierge-agent"
}

variable "container_image" {
  description = "Artifact Registry container image URI for the Meal Concierge ADK service."
  type        = string
  default     = "us-east1-docker.pkg.dev/agiljohn-cert-oct2026/agents/meal-concierge:latest"
}

variable "enable_deletion_protection" {
  description = "Whether to enable deletion protection on Cloud Run and storage resources."
  type        = bool
  default     = true
}
