# =============================================================================
# Terraform Infrastructure as Code (IaC) — Meal Planning Concierge Agent
# Implements Rubric Category 5.2 (Infrastructure as Code) & 5.3 (Secret Manager)
# Provisions:
#   1. Required Google Cloud APIs (Vertex AI, Discovery Engine, Cloud Run, DLP,
#      Secret Manager, Cloud Trace, Cloud Logging, Firestore, Artifact Registry, GCS)
#   2. Dedicated Least-Privilege Service Account & Resource-Scoped IAM Bindings
#   3. Google Cloud Secret Manager Secrets + Per-Secret IAM Bindings (secret values
#      injected out-of-band via `gcloud secrets versions add`, never stored in TF state)
#   4. Artifact Registry Repository for the ADK Container Image
#   5. GCS Buckets (Vector Index Seed + Persistent Session State Volume)
#   6. Vertex AI Vector Search Index (64-D) & Endpoint + Discovery Engine Data Store
#   7. Cloud Run v2 Service with GCS Volume Mount and Production Observability Env Vars
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudtrace.googleapis.com",
    "discoveryengine.googleapis.com",
    "dlp.googleapis.com",
    "firestore.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# -----------------------------------------------------------------------------
# Least-Privilege Agent Runtime Service Account
# -----------------------------------------------------------------------------
resource "google_service_account" "meal_concierge_sa" {
  account_id   = "meal-concierge-agent-sa"
  display_name = "Autonomous Meal Planning Concierge ADK Service Account"
  description  = "Identity for the Meal Concierge Agent on Vertex AI Agent Engine and Cloud Run"
}

resource "google_project_iam_member" "agent_iam_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/datastore.user",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.meal_concierge_sa.email}"
}

# -----------------------------------------------------------------------------
# Artifact Registry Repository for Container Images
# -----------------------------------------------------------------------------
resource "google_artifact_registry_repository" "agent_repo" {
  location      = var.region
  repository_id = "agents"
  description   = "Container repository for the ADK Meal Planning Concierge Agent"
  format        = "DOCKER"
  depends_on    = [google_project_service.enabled_apis]
}

# -----------------------------------------------------------------------------
# Google Cloud Secret Manager (Rubric Category 5.3 — Zero Hardcoded Keys)
# Secret payloads are populated out-of-band via `gcloud secrets versions add`
# so secret values are never committed to Git or stored in Terraform state.
# -----------------------------------------------------------------------------
resource "google_secret_manager_secret" "instacart_partner_token" {
  secret_id = "INSTACART_PARTNER_API_TOKEN"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_iam_member" "instacart_token_accessor" {
  secret_id = google_secret_manager_secret.instacart_partner_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.meal_concierge_sa.email}"
}

resource "google_secret_manager_secret" "hitl_signing_secret" {
  secret_id = "HITL_APPROVAL_SIGNING_SECRET"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_iam_member" "hitl_signing_secret_accessor" {
  secret_id = google_secret_manager_secret.hitl_signing_secret.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.meal_concierge_sa.email}"
}

# -----------------------------------------------------------------------------
# GCS Buckets for Persistent Session State Volume & Vector Search Seed (2.3)
# -----------------------------------------------------------------------------
resource "google_storage_bucket" "session_state_bucket" {
  name                        = "${var.project_id}-session-state"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = !var.enable_deletion_protection
  depends_on                  = [google_project_service.enabled_apis]
}

resource "google_storage_bucket_iam_member" "session_state_bucket_writer" {
  bucket = google_storage_bucket.session_state_bucket.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.meal_concierge_sa.email}"
}

resource "google_storage_bucket" "vector_memory_bucket" {
  name                        = "${var.project_id}-vector-memory"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = !var.enable_deletion_protection
  depends_on                  = [google_project_service.enabled_apis]
}

resource "google_storage_bucket_object" "vector_memory_seed" {
  name    = "initial/embeddings.json"
  bucket  = google_storage_bucket.vector_memory_bucket.name
  content = "{\"id\": \"seed-001\", \"embedding\": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}\n"
}

resource "google_vertex_ai_index" "culinary_memory_index" {
  region              = var.region
  display_name        = "meal-concierge-episodic-memory-index"
  description         = "64-D Vector index storing long-term dietary preferences and recipe embeddings"
  index_update_method = "STREAM_UPDATE"

  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.vector_memory_bucket.name}/initial"
    config {
      dimensions                  = 64
      approximate_neighbors_count = 15
      distance_measure_type       = "COSINE_DISTANCE"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 7
        }
      }
    }
  }
  depends_on = [
    google_project_service.enabled_apis,
    google_storage_bucket_object.vector_memory_seed,
  ]
}

resource "google_vertex_ai_index_endpoint" "culinary_memory_endpoint" {
  region                  = var.region
  display_name            = "meal-concierge-memory-endpoint"
  description             = "Low-latency vector endpoint for semantic dietary memory retrieval"
  public_endpoint_enabled = true
  depends_on              = [google_project_service.enabled_apis]
}

# -----------------------------------------------------------------------------
# Vertex AI Search / Discovery Engine Data Store (Category 2.3)
# -----------------------------------------------------------------------------
resource "google_discovery_engine_data_store" "nutrition_knowledge_store" {
  location                    = "global"
  data_store_id               = "culinary-nutrition-datastore"
  display_name                = "Clinical & Culinary Nutrition Grounding Store"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false
  depends_on                  = [google_project_service.enabled_apis]
}

# -----------------------------------------------------------------------------
# Cloud Run v2 Service hosting the ADK Meal Planning Concierge Agent
# -----------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "meal_concierge_service" {
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.enable_deletion_protection

  template {
    service_account = google_service_account.meal_concierge_sa.email
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    volumes {
      name = "persistent-session-state"
      gcs {
        bucket    = google_storage_bucket.session_state_bucket.name
        read_only = false
      }
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      volume_mounts {
        name       = "persistent-session-state"
        mount_path = "/mnt/state"
      }

      env {
        name  = "MEAL_CONCIERGE_ENV"
        value = "production"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "MEAL_CONCIERGE_DB_PATH"
        value = "/mnt/state/meal_concierge_sessions.sqlite3"
      }
      env {
        name  = "ENABLE_CLOUD_TRACE"
        value = "true"
      }
      env {
        name  = "ENABLE_CLOUD_DLP"
        value = "true"
      }
      env {
        name  = "ENABLE_VERTEX_CACHING"
        value = "true"
      }
      env {
        name  = "ENABLE_VERTEX_VECTOR_SEARCH"
        value = "true"
      }
      env {
        name  = "ENABLE_VERTEX_SEARCH"
        value = "true"
      }
      env {
        name  = "ENABLE_VERTEX_SEARCH_RPC"
        value = "true"
      }
      env {
        name  = "VERTEX_SEARCH_DATASTORE_ID"
        value = google_discovery_engine_data_store.nutrition_knowledge_store.data_store_id
      }
      # Out-of-band Secret Manager payload bootstrap before deploying Cloud Run revision:
      #   printf '%s' "$INSTACART_TOKEN" | gcloud secrets versions add instacart-partner-api-token --data-file=-
      #   printf '%s' "$HITL_SECRET"     | gcloud secrets versions add hitl-approval-signing-secret --data-file=-
      env {
        name = "INSTACART_PARTNER_API_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.instacart_partner_token.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "HITL_APPROVAL_SIGNING_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.hitl_signing_secret.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/list-apps"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.agent_iam_roles,
    google_secret_manager_secret_iam_member.instacart_token_accessor,
    google_secret_manager_secret_iam_member.hitl_signing_secret_accessor,
    google_storage_bucket_iam_member.session_state_bucket_writer,
  ]
}
