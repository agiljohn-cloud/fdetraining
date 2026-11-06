"""Deploy ChefAura Meal Planning Concierge Agent to Vertex AI Agent Platform (ReasoningEngine).

Targets:
- GCP Project: `agiljohn-cert-oct2026`
- Location:    `us-east1`
- Staging GCS: `gs://agiljohn-cert-oct2026-agent-staging-useast1`

Follows Google Cloud Vertex AI Agent Engine (`client.agent_engines.create` / `ReasoningEngine`)
and Agent Registry best practices:
- Wraps `root_agent` in an `AdkApp` subclass with `enable_tracing=True` and a generator
  `stream_query` (`yield from super().stream_query(...)`) for streaming and Agent Registry support.
- Reuses an existing deployment in `agiljohn-cert-oct2026` (`us-east1`) if one already exists
  before creating a new `ReasoningEngine`.
- Writes `deployment_metadata.json` with the live `projects/.../locations/us-east1/reasoningEngines/...`
  resource name and Pantheon Agent Registry console URL.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

DEFAULT_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "agiljohn-cert-oct2026")
DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")
DEFAULT_DISPLAY_NAME = "meal-concierge-agent"
DEFAULT_STAGING_BUCKET = f"gs://{DEFAULT_PROJECT_ID}-agent-staging-useast1"
PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_ROOT / "deployment_metadata.json"

DEPLOY_REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]>=1.88.0",
    "google-adk>=1.0.0",
    "google-genai>=1.10.0",
    "pydantic>=2.7.0",
    "cloudpickle>=3.0.0",
    "opentelemetry-api>=1.25.0",
    "opentelemetry-sdk>=1.25.0",
]


def get_gcloud_user_credentials(project_id: str) -> Any:
    """Build a refreshable `google.oauth2.credentials.Credentials` backed by `gcloud auth print-access-token`."""
    import google.auth
    from google.oauth2.credentials import Credentials

    def _fetch_token() -> str:
        out = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
        ).strip()
        if not out:
            raise RuntimeError("Empty token returned by `gcloud auth print-access-token`.")
        return out

    class _GcloudCliCredentials(Credentials):
        def __init__(self, token: str, quota_project_id: str) -> None:
            super().__init__(token=token, quota_project_id=quota_project_id)

        def refresh(self, request: Any) -> None:
            self.token = _fetch_token()

        @property
        def valid(self) -> bool:
            return bool(self.token)

        @property
        def expired(self) -> bool:
            return False

    creds = _GcloudCliCredentials(token=_fetch_token(), quota_project_id=project_id)
    # Ensure internal SDK calls (e.g. google.cloud.storage.Client during staging upload)
    # also use the authenticated gcloud CLI user credentials rather than GCE metadata server.
    google.auth.default = lambda *args, **kwargs: (creds, project_id)  # type: ignore
    return creds


def ensure_apis_and_staging_bucket(project_id: str, location: str, staging_bucket: str) -> None:
    """Ensure required Vertex AI / Agent Platform APIs and the regional GCS staging bucket exist."""
    apis = [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "secretmanager.googleapis.com",
        "cloudtrace.googleapis.com",
        "logging.googleapis.com",
        "discoveryengine.googleapis.com",
        "dlp.googleapis.com",
        "run.googleapis.com",
        "cloudbuild.googleapis.com",
    ]
    print(f"[1/4] Enabling required Google Cloud APIs in project '{project_id}'...")
    subprocess.run(
        ["gcloud", "services", "enable", *apis, f"--project={project_id}", "--quiet"],
        check=True,
    )

    print(f"[2/4] Ensuring staging bucket '{staging_bucket}' exists in '{location}'...")
    check_res = subprocess.run(
        ["gcloud", "storage", "buckets", "describe", staging_bucket, f"--project={project_id}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check_res.returncode != 0:
        subprocess.run(
            [
                "gcloud",
                "storage",
                "buckets",
                "create",
                staging_bucket,
                f"--project={project_id}",
                f"--location={location}",
                "--uniform-bucket-level-access",
                "--quiet",
            ],
            check=True,
        )


def build_adk_app(
    project_id: str = DEFAULT_PROJECT_ID,
    location: str = DEFAULT_LOCATION,
    credentials: Any = None,
) -> Any:
    """Build the `AdkApp` wrapper around `meal_concierge.agent` with streaming enabled."""
    import vertexai
    from meal_concierge.agent import build_meal_concierge_agent_hierarchy

    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["CHEFAURA_PRO_MODEL"] = "gemini-3.1-pro-preview"
    os.environ["CHEFAURA_FLASH_MODEL"] = "gemini-3.5-flash"
    vertexai.init(project=project_id, location=location, credentials=credentials)

    try:
        from vertexai.agent_engines import AdkApp
    except ImportError:
        from vertexai.preview.reasoning_engines import AdkApp  # type: ignore

    runtime_agent = build_meal_concierge_agent_hierarchy()

    class ChefAuraAdkApp(AdkApp):
        """ADK App wrapper routing Gemini 3.1 Pro / 3.5 Flash calls to the global Vertex endpoint."""

        def set_up(self) -> None:
            os.environ["CHEFAURA_PRO_MODEL"] = "gemini-3.1-pro-preview"
            os.environ["CHEFAURA_FLASH_MODEL"] = "gemini-3.5-flash"
            super().set_up()
            # Route Gemini 3.1 Pro / 3.5 Flash publisher model invocations to the Vertex AI global endpoint
            os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

    return ChefAuraAdkApp(agent=runtime_agent)


def find_existing_engine_resource_name(
    project_id: str,
    location: str,
    display_name: str,
    credentials: Any = None,
) -> Optional[str]:
    """Check `deployment_metadata.json` or list existing ReasoningEngines in the target project/region."""
    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            candidate = str(meta.get("remote_agent_runtime_id", ""))
            if f"/locations/{location}/reasoningEngines/" in candidate and (
                candidate.startswith(f"projects/{project_id}/")
                or meta.get("project_id") == project_id
            ):
                return candidate
        except Exception:
            pass

    try:
        import vertexai

        client = vertexai.Client(project=project_id, location=location, credentials=credentials)
        for engine in client.agent_engines.list():
            api_res = getattr(engine, "api_resource", engine)
            if getattr(api_res, "display_name", "") == display_name:
                return str(getattr(api_res, "name", ""))
    except Exception:
        pass
    return None


def deploy_to_agent_platform(
    project_id: str = DEFAULT_PROJECT_ID,
    location: str = DEFAULT_LOCATION,
    staging_bucket: str = DEFAULT_STAGING_BUCKET,
    display_name: str = DEFAULT_DISPLAY_NAME,
) -> Dict[str, Any]:
    """Deploy or update the ADK Meal Planning Concierge Agent on Vertex AI Agent Platform."""
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    creds = get_gcloud_user_credentials(project_id)
    if not METADATA_PATH.exists():
        ensure_apis_and_staging_bucket(project_id, location, staging_bucket)

    import vertexai

    os.chdir(PROJECT_ROOT)
    adk_app = build_adk_app(project_id=project_id, location=location, credentials=creds)

    print(
        f"[3/4] Deploying '{display_name}' to Vertex AI Agent Platform in {project_id} ({location})..."
    )
    config = {
        "display_name": display_name,
        "description": (
            "ChefAura — Autonomous Multi-Agent Meal Planning Concierge built on Google ADK "
            "(Mifflin-St Jeor macro calculator, parallel pantry auditor, allergen-safe recipe "
            "architect, and HITL-gated grocery procurement specialist)."
        ),
        "staging_bucket": staging_bucket,
        "requirements": DEPLOY_REQUIREMENTS,
        "extra_packages": ["meal_concierge"],
        "env_vars": {
            "CHEFAURA_GCP_PROJECT": project_id,
            "CHEFAURA_GCP_LOCATION": location,
            "CHEFAURA_PRO_MODEL": "gemini-3.1-pro-preview",
            "CHEFAURA_FLASH_MODEL": "gemini-3.5-flash",
            "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
            "ENABLE_CLOUD_TRACE": "true",
        },
    }

    resource_name: Optional[str] = None
    existing_name = find_existing_engine_resource_name(
        project_id, location, display_name, credentials=creds
    )

    if hasattr(vertexai, "Client"):
        client = vertexai.Client(project=project_id, location=location, credentials=creds)
        if existing_name:
            print(f"Reusing and updating existing ReasoningEngine: {existing_name}")
            remote_agent = client.agent_engines.update(
                name=existing_name,
                agent=adk_app,
                config=config,
            )
        else:
            remote_agent = client.agent_engines.create(
                agent=adk_app,
                config=config,
            )
        api_res = getattr(remote_agent, "api_resource", remote_agent)
        resource_name = str(
            getattr(api_res, "name", None) or getattr(remote_agent, "resource_name", "")
        )
    else:
        from vertexai.preview.reasoning_engines import ReasoningEngine

        vertexai.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket,
            credentials=creds,
        )
        remote_agent = ReasoningEngine.create(
            reasoning_engine=adk_app,
            requirements=DEPLOY_REQUIREMENTS,
            extra_packages=["meal_concierge"],
            display_name=display_name,
            description=config["description"],
        )
        resource_name = str(remote_agent.resource_name)

    engine_id = resource_name.split("/")[-1] if resource_name else ""
    pantheon_agent_engine_url = (
        f"https://pantheon.corp.google.com/vertex-ai/agents/agent-engines/"
        f"locations/{location}/agent-engines/{engine_id}?project={project_id}"
    )
    pantheon_registry_url = (
        f"https://pantheon.corp.google.com/agent-management/agent-registry/agents"
        f"?project={project_id}"
    )

    metadata = {
        "remote_agent_runtime_id": resource_name,
        "engine_id": engine_id,
        "project_id": project_id,
        "location": location,
        "staging_bucket": staging_bucket,
        "display_name": display_name,
        "deployment_target": "agent_runtime",
        "framework": "google-adk",
        "pantheon_agent_engine_url": pantheon_agent_engine_url,
        "pantheon_agent_registry_url": pantheon_registry_url,
        "deployment_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] Deployment succeeded! Resource name: {resource_name}")
    print(f"      Pantheon Agent Engine URL: {pantheon_agent_engine_url}")
    return metadata


def query_deployed_agent(
    prompt: str,
    user_id: str = "agiljohn",
    project_id: str = DEFAULT_PROJECT_ID,
    location: str = DEFAULT_LOCATION,
    resource_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a live query to the deployed ReasoningEngine on Vertex AI Agent Platform."""
    creds = get_gcloud_user_credentials(project_id)
    resolved_name = resource_name or find_existing_engine_resource_name(
        project_id, location, DEFAULT_DISPLAY_NAME, credentials=creds
    )
    if not resolved_name:
        raise RuntimeError(
            "No deployed ReasoningEngine resource found. Run `python3 deploy_agent_platform.py` first."
        )

    import vertexai

    client = vertexai.Client(project=project_id, location=location, credentials=creds)
    remote_agent = client.agent_engines.get(name=resolved_name)
    events = []
    if hasattr(remote_agent, "stream_query"):
        for event in remote_agent.stream_query(user_id=user_id, message=prompt):
            events.append(event)
    elif hasattr(remote_agent, "query"):
        events.append(remote_agent.query(input=prompt, user_id=user_id))

    return {
        "status": "success",
        "resource_name": resolved_name,
        "project_id": project_id,
        "location": location,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy or query ChefAura Meal Planning Concierge on Vertex AI Agent Platform."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT_ID, help="GCP Project ID")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="GCP Region (default: us-east1)")
    parser.add_argument(
        "--staging-bucket",
        default=None,
        help="GCS staging bucket URI (default: gs://<project>-agent-staging-useast1)",
    )
    parser.add_argument("--query", default=None, help="Optional prompt to send to the deployed agent")
    args = parser.parse_args()

    bucket = args.staging_bucket or f"gs://{args.project}-agent-staging-useast1"
    if args.query:
        res = query_deployed_agent(
            prompt=args.query,
            project_id=args.project,
            location=args.location,
        )
        print(json.dumps(res, indent=2, default=str))
    else:
        meta = deploy_to_agent_platform(
            project_id=args.project,
            location=args.location,
            staging_bucket=bucket,
        )
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
