output "cloud_run_service_uri" {
  description = "Public HTTPS endpoint of the deployed Meal Planning Concierge Agent."
  value       = google_cloud_run_v2_service.meal_concierge_service.uri
}

output "agent_service_account_email" {
  description = "Least-privilege service account email bound to the agent runtime."
  value       = google_service_account.meal_concierge_sa.email
}

output "vertex_vector_index_id" {
  description = "Vertex AI Vector Search Index resource ID for episodic culinary memory."
  value       = google_vertex_ai_index.culinary_memory_index.id
}

output "vertex_search_datastore_id" {
  description = "Vertex AI Discovery Engine Data Store ID for clinical nutrition grounding."
  value       = google_discovery_engine_data_store.nutrition_knowledge_store.data_store_id
}
