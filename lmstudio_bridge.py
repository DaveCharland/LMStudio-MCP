#!/usr/bin/env python3
"""
LM Studio MCP Bridge
Provides Claude Code access to LM Studio models on multiple devices.
Supports M3 Ultra (remote) and M4 Max (local).
"""

from mcp.server.fastmcp import FastMCP
import requests
import json
import sys
import os
import asyncio
from typing import List, Dict, Any, Optional

# Initialize FastMCP server
mcp = FastMCP("lmstudio-bridge")

# Named hosts for easy reference
KNOWN_HOSTS = {
    "m3": "http://192.168.1.206:1234",      # M3 Ultra (remote)
    "m3-ultra": "http://192.168.1.206:1234",
    "m4": "http://localhost:1234",           # M4 Max (local)
    "m4-max": "http://localhost:1234",
    "local": "http://localhost:1234",
    "remote": "http://192.168.1.206:1234",
}

# Default host - can be overridden by LMSTUDIO_HOST env var
DEFAULT_HOST = os.environ.get("LMSTUDIO_HOST", "http://192.168.1.206:1234")


def resolve_host(host: str = "") -> str:
    """Resolve a host parameter to a full URL."""
    if not host:
        return DEFAULT_HOST
    if host in KNOWN_HOSTS:
        return KNOWN_HOSTS[host]
    if host.startswith("http"):
        return host
    return f"http://{host}:1234"


def log_error(message: str):
    """Log error messages to stderr for debugging"""
    print(f"ERROR: {message}", file=sys.stderr)


def log_info(message: str):
    """Log informational messages to stderr for debugging"""
    print(f"INFO: {message}", file=sys.stderr)


@mcp.tool()
async def health_check(host: str = "") -> str:
    """Check if LM Studio API is accessible on specified host.

    Args:
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.

    Returns:
        A message indicating whether the LM Studio API is running.
    """
    base_url = resolve_host(host)
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json().get("data", [])
            model_names = [m.get("id", "unknown") for m in models]
            return f"LM Studio at {base_url} is running.\nLoaded models: {', '.join(model_names) if model_names else 'None'}"
        else:
            return f"LM Studio at {base_url} returned status code {response.status_code}."
    except Exception as e:
        return f"Error connecting to LM Studio at {base_url}: {str(e)}"


@mcp.tool()
async def list_models(host: str = "") -> str:
    """List all available models in LM Studio.

    Args:
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.

    Returns:
        A formatted list of available models.
    """
    base_url = resolve_host(host)
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code != 200:
            return f"Failed to fetch models from {base_url}. Status code: {response.status_code}"

        models = response.json().get("data", [])
        if not models:
            return f"No models loaded in LM Studio at {base_url}."

        result = f"Available models at {base_url}:\n\n"
        for model in models:
            result += f"- {model['id']}\n"

        return result
    except Exception as e:
        log_error(f"Error in list_models: {str(e)}")
        return f"Error listing models at {base_url}: {str(e)}"


@mcp.tool()
async def get_current_model(host: str = "") -> str:
    """Get the currently loaded model in LM Studio.

    Args:
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.

    Returns:
        The name of the currently loaded model.
    """
    base_url = resolve_host(host)
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "system", "content": "What model are you?"}],
                "temperature": 0.7,
                "max_tokens": 10
            },
            timeout=30
        )

        if response.status_code != 200:
            return f"Failed to identify model at {base_url}. Status code: {response.status_code}"

        model_info = response.json().get("model", "Unknown")
        return f"Currently loaded model at {base_url}: {model_info}"
    except Exception as e:
        log_error(f"Error in get_current_model: {str(e)}")
        return f"Error identifying model at {base_url}: {str(e)}"


@mcp.tool()
async def load_model(model_name: str, host: str = "") -> str:
    """Load a specific model in LM Studio.

    Args:
        model_name: The name of the model to load (e.g., 'deepseek-r1-distill-llama-70b')
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.

    Returns:
        Status message indicating success or failure.
    """
    base_url = resolve_host(host)
    try:
        log_info(f"Loading model '{model_name}' on {base_url}")

        response = requests.post(
            f"{base_url}/api/v1/models/load",
            json={"model": model_name},
            timeout=300  # 5 minute timeout for large models
        )

        if response.status_code == 200:
            result = response.json()
            load_time = result.get("load_time_seconds", "unknown")
            return f"Model '{model_name}' loaded successfully on {base_url} in {load_time} seconds."
        else:
            return f"Failed to load model on {base_url}. Status: {response.status_code}. Response: {response.text}"
    except requests.exceptions.Timeout:
        return f"Timeout loading '{model_name}' on {base_url}. Model may still be loading - check LM Studio."
    except Exception as e:
        log_error(f"Error in load_model: {str(e)}")
        return f"Error loading model on {base_url}: {str(e)}"


@mcp.tool()
async def unload_model(instance_id: str, host: str = "") -> str:
    """Unload a model from LM Studio.

    Args:
        instance_id: The instance ID of the model to unload (usually same as model name)
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.

    Returns:
        Status message indicating success or failure.
    """
    base_url = resolve_host(host)
    try:
        log_info(f"Unloading model '{instance_id}' from {base_url}")

        response = requests.post(
            f"{base_url}/api/v1/models/unload",
            json={"instance_id": instance_id},
            timeout=60
        )

        if response.status_code == 200:
            return f"Model '{instance_id}' unloaded successfully from {base_url}."
        else:
            return f"Failed to unload model from {base_url}. Status: {response.status_code}. Response: {response.text}"
    except Exception as e:
        log_error(f"Error in unload_model: {str(e)}")
        return f"Error unloading model from {base_url}: {str(e)}"


@mcp.tool()
async def chat_completion(
    prompt: str,
    host: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:
    """Generate a completion from an LM Studio model.

    Args:
        prompt: The user's prompt to send to the model
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.
        system_prompt: Optional system instructions for the model
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate

    Returns:
        The model's response to the prompt
    """
    base_url = resolve_host(host)
    try:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        log_info(f"Sending request to {base_url} with {len(messages)} messages")

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=120
        )

        if response.status_code != 200:
            log_error(f"LM Studio API error: {response.status_code}")
            return f"Error: LM Studio at {base_url} returned status code {response.status_code}"

        response_json = response.json()
        model_used = response_json.get("model", "unknown")
        log_info(f"Received response from {model_used} at {base_url}")

        choices = response_json.get("choices", [])
        if not choices:
            return "Error: No response generated"

        content = choices[0].get("message", {}).get("content", "")

        if not content:
            return "Error: Empty response from model"

        # Append model info
        content += f"\n\n---\n_Model: {model_used} | Host: {base_url}_"

        return content
    except requests.exceptions.Timeout:
        return f"Error: Request to {base_url} timed out after 120 seconds"
    except Exception as e:
        log_error(f"Error in chat_completion: {str(e)}")
        return f"Error generating completion from {base_url}: {str(e)}"


@mcp.tool()
async def list_hosts() -> str:
    """List available LM Studio host shortcuts.

    Returns:
        A list of host shortcuts and their URLs.
    """
    result = "Available LM Studio Hosts:\n\n"
    result += "| Shortcut    | URL                          | Description      |\n"
    result += "|-------------|------------------------------|------------------|\n"
    result += "| m3, m3-ultra| http://192.168.1.206:1234    | M3 Ultra (remote)|\n"
    result += "| m4, m4-max  | http://localhost:1234        | M4 Max (local)   |\n"
    result += "| local       | http://localhost:1234        | Same as m4       |\n"
    result += "| remote      | http://192.168.1.206:1234    | Same as m3       |\n"
    result += f"\nDefault host: {DEFAULT_HOST}\n"
    result += "\nUsage: chat_completion(prompt='Hello', host='m3')"
    return result


async def _process_single_prompt(
    base_url: str,
    prompt_item: dict,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process single prompt with concurrency control."""
    prompt_id = prompt_item.get("id", "prompt_0")
    prompt_text = prompt_item.get("prompt", str(prompt_item))

    async with semaphore:
        try:
            def do_request():
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt_text})

                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=120
                )
                return response

            response = await asyncio.to_thread(do_request)

            if response.status_code == 200:
                data = response.json()
                return {
                    "id": prompt_id,
                    "status": "success",
                    "content": data["choices"][0]["message"]["content"],
                    "model": data.get("model", "unknown"),
                    "tokens": data.get("usage", {})
                }
            else:
                return {
                    "id": prompt_id,
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "content": None
                }

        except requests.exceptions.Timeout:
            return {
                "id": prompt_id,
                "status": "error",
                "error": "timeout (120s)",
                "content": None
            }
        except Exception as e:
            return {
                "id": prompt_id,
                "status": "error",
                "error": str(e),
                "content": None
            }


# Batch processing guardrails (empirically tested 2026-02-02)
# These limits ensure reliable operation without connection drops
BATCH_GUARDRAILS = {
    "max_prompts": 15,           # Hard limit - 20+ causes failures
    "recommended_prompts": 12,   # Sweet spot for reliability
    "default_concurrency": 4,    # Most stable setting
    "max_concurrency": 6,        # Higher causes connection drops
    "max_tokens_conservative": 100,  # For 12+ prompts
    "max_tokens_standard": 150,      # For 8-11 prompts
    "max_tokens_aggressive": 200,    # For ≤7 prompts
}


@mcp.tool()
async def batch_completion(
    prompts: str,
    host: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 100,
    concurrency: int = 4
) -> str:
    """Process multiple prompts in parallel on LM Studio.

    GUARDRAILS (tested 2026-02-02 on M3 Ultra):
    - Max 15 prompts per batch (12 recommended for reliability)
    - Concurrency 4 is optimal (max 6, higher causes drops)
    - For 12+ prompts: keep max_tokens ≤100
    - For 8-11 prompts: max_tokens ≤150
    - For ≤7 prompts: max_tokens ≤200
    - Do NOT call multiple batch_completion in parallel

    Args:
        prompts: JSON array of prompts. Either simple strings or objects with id/prompt keys.
                 Examples: '["prompt1", "prompt2"]' or '[{"id": "a", "prompt": "text"}]'
        host: Host shortname (m3, m4, local, remote) or URL. Default: M3 Ultra.
        system_prompt: Optional system instructions applied to all prompts.
        temperature: Controls randomness (0.0 to 1.0).
        max_tokens: Maximum tokens per response. Default: 100 (conservative).
        concurrency: Max parallel requests (1-6). Default: 4.

    Returns:
        JSON with summary and results array including success/failure status for each prompt.
    """
    base_url = resolve_host(host)

    # Parse prompts JSON
    try:
        prompt_list = json.loads(prompts)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"}, indent=2)

    if not isinstance(prompt_list, list):
        return json.dumps({"error": "prompts must be a JSON array"}, indent=2)

    if len(prompt_list) == 0:
        return json.dumps({"error": "prompts array is empty"}, indent=2)

    # Enforce guardrails
    warnings = []

    if len(prompt_list) > BATCH_GUARDRAILS["max_prompts"]:
        return json.dumps({
            "error": f"Too many prompts: {len(prompt_list)}. Maximum is {BATCH_GUARDRAILS['max_prompts']}. Split into multiple sequential batches.",
            "guardrails": BATCH_GUARDRAILS
        }, indent=2)

    if len(prompt_list) > BATCH_GUARDRAILS["recommended_prompts"]:
        warnings.append(f"Prompt count ({len(prompt_list)}) exceeds recommended ({BATCH_GUARDRAILS['recommended_prompts']}). Some requests may fail.")

    # Normalize to dict format with id and prompt keys
    normalized = []
    for i, p in enumerate(prompt_list):
        if isinstance(p, str):
            normalized.append({"id": f"prompt_{i}", "prompt": p})
        elif isinstance(p, dict) and "prompt" in p:
            normalized.append({"id": p.get("id", f"prompt_{i}"), "prompt": p["prompt"]})
        else:
            normalized.append({
                "id": f"prompt_{i}",
                "prompt": str(p) if p else ""
            })

    # Apply concurrency guardrails (1-6, not 1-8)
    actual_concurrency = min(max(1, concurrency), BATCH_GUARDRAILS["max_concurrency"])
    if concurrency > BATCH_GUARDRAILS["max_concurrency"]:
        warnings.append(f"Concurrency clamped from {concurrency} to {actual_concurrency} (max stable)")

    # Warn about risky max_tokens combinations
    if len(normalized) >= 12 and max_tokens > BATCH_GUARDRAILS["max_tokens_conservative"]:
        warnings.append(f"High max_tokens ({max_tokens}) with {len(normalized)} prompts may cause failures. Recommended: ≤{BATCH_GUARDRAILS['max_tokens_conservative']}")

    log_info(f"Batch processing {len(normalized)} prompts on {base_url} (concurrency={actual_concurrency})")

    # Execute in parallel with semaphore for rate limiting
    semaphore = asyncio.Semaphore(actual_concurrency)
    tasks = [
        _process_single_prompt(base_url, p, system_prompt, temperature, max_tokens, semaphore)
        for p in normalized
    ]
    results = await asyncio.gather(*tasks)

    # Build response summary
    successful = sum(1 for r in results if r["status"] == "success")

    response = {
        "summary": {
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "host": base_url,
            "concurrency": actual_concurrency
        },
        "results": results
    }

    # Include warnings if any
    if warnings:
        response["warnings"] = warnings

    return json.dumps(response, indent=2)


def main():
    """Entry point for the package when installed via pip"""
    log_info("Starting LM Studio Bridge MCP Server (Multi-Host)")
    log_info(f"Default host: {DEFAULT_HOST}")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
