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
) -> Any:
    """Build the `AdkApp` wrapper around `meal_concierge.agent.root_agent` with streaming enabled."""
    import vertexai
    from meal_concierge.agent import root_agent

    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    vertexai.init(project=project_id, location=location)

    try:
        from vertexai.agent_engines import AdkApp
    except ImportError:
        from vertexai.preview.reasoning_engines import AdkApp  # type: ignore

    class ChefAuraAdkApp(AdkApp):
        """ADK App wrapper supporting both synchronous `query` and generator `stream_query`."""

        def query(self, *args: Any, **kwargs: Any) -> Any:
            return super().query(*args, **kwargs)

        def stream_query(self, *args: Any, **kwargs: Any):
            yield from super().stream_query(*args, **kwargs)

    return ChefAuraAdkApp(agent=root_agent, enable_tracing=True)


def find_existing_engine_resource_name(
    project_id: str,
    location: str,
    display_name: str,
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

        client = vertexai.Client(project=project_id, location=location)
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

    ensure_apis_and_staging_bucket(project_id, location, staging_bucket)

    import vertexai

    os.chdir(PROJECT_ROOT)
    adk_app = build_adk_app()

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
            "GOOGLE_CLOUD_PROJECT": project_id,
            "GOOGLE_CLOUD_LOCATION": location,
            "ENABLE_CLOUD_TRACE": "true",
        },
    }

    resource_name: Optional[str] = None
    existing_name = find_existing_engine_resource_name(project_id, location, display_name)

    if hasattr(vertexai, "Client"):
        client = vertexai.Client(project=project_id, location=location)
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

        vertexai.init(project=project_id, location=location, staging_bucket=staging_bucket)
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
    resolved_name = resource_name or find_existing_engine_resource_name(
        project_id, location, DEFAULT_DISPLAY_NAME
    )
    if not resolved_name:
        raise RuntimeError(
            "No deployed ReasoningEngine resource found. Run `python3 deploy_agent_platform.py` first."
        )

    import vertexai

    events = []
    if hasattr(vertexai, "Client"):
        client = vertexai.Client(project=project_id, location=location)
        remote_agent = client.agent_engines.get(name=resolved_name)
        for event in remote_agent.stream_query(user_id=user_id, message=prompt):
            events.append(event)
    else:
        from vertexai.preview.reasoning_engines import ReasoningEngine

        vertexai.init(project=project_id, location=location)
        remote_agent = ReasoningEngine(resolved_name)
        for event in remote_agent.stream_query(user_id=user_id, message=prompt):
            events.append(event)

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
