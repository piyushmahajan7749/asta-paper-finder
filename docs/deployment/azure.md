# Deploying PaperFinder to Azure App Service (Container)

This guide describes how to ship the FastAPI server (`/agents/mabool/api`) to Azure using a container image. You will end up with a stable `https://<app-name>.azurewebsites.net` URL that your frontend can call.

## 0. Prerequisites

- Azure CLI logged in (`az login`).
- An Azure resource group (use `az group create` if you do not have one).
- Docker and access to the repo root (where the `Dockerfile` lives).
- All required API keys handy (`OPENAI_API_KEY` for Azure OpenAI, `S2_API_KEY`, `COHERE_API_KEY`). (`GOOGLE_API_KEY` is only needed if you keep `google:*` models enabled.)

## 1. Build and test the image locally

> ⚠️ **Architecture warning for Apple Silicon (M1/M2/M3) Macs.** Azure App Service
> runs on `linux/amd64`. A plain `docker build` on an ARM Mac produces a
> `linux/arm64` image — it'll push to ACR fine, but the App Service container
> will fail to start with `exec /app/.venv/bin/uvicorn: exec format error` in
> the docker log (the binary is the wrong architecture for the host kernel,
> and Linux refuses to `exec` it). Always pass `--platform linux/amd64` when
> building for Azure deploy.

```bash
cd /path/to/asta-paper-finder

# Cross-build for Azure's amd64 runtime (works on both Intel and Apple Silicon Macs)
docker buildx build --platform linux/amd64 -t paperfinder-api .

# Smoke-test locally. On Apple Silicon this runs under Rosetta-style emulation
# (slower than native), which is fine for the health check.
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=... \
  -e AZURE_OPENAI_ENDPOINT=... \
  -e AZURE_OPENAI_API_VERSION=... \
  -e AZURE_OPENAI_DEPLOYMENT=... \
  -e S2_API_KEY=... \
  -e COHERE_API_KEY=... \
  -e OPENALEX_MAILTO=ops@yourdomain.com \
  paperfinder-api
curl http://localhost:8000/health   # should return 204
```

## 2. Push the image to Azure Container Registry (ACR)

```bash
# Create a registry (skip if you already have one)
az acr create --name <acr-name> --resource-group <rg> --sku Basic

az acr login --name <acr-name>
docker tag paperfinder-api <acr-name>.azurecr.io/paperfinder-api:v1
docker push <acr-name>.azurecr.io/paperfinder-api:v1
```

> **Tip:** to bake the `--platform` flag into your build instead of remembering
> it every time, you can run `docker buildx build --platform linux/amd64
> --push -t <acr-name>.azurecr.io/paperfinder-api:v1 .` — buildx will build
> AND push the amd64 image in one shot, skipping the separate `docker push`.

## 3. Provision the App Service

```bash
az appservice plan create \
  --name paperfinder-plan \
  --resource-group roofaiopenai \
  --is-linux \
  --sku B2

az webapp create \
  --name paperfinder-api \
  --resource-group roofaiopenai \
  --plan paperfinder-plan \
  --deployment-container-image-name <acr-name>.azurecr.io/paperfinder-api:v1
```

Tell the Web App how to pull from your private registry (skip `--docker-registry-server-*` if the registry is public):

```bash
az webapp config container set \
  --name paperfinder-api \
  --resource-group roofaiopenai \
  --docker-custom-image-name <acr-name>.azurecr.io/paperfinder-api:v1 \
  --docker-registry-server-url https://<acr-name>.azurecr.io \
  --docker-registry-server-user <acr-username> \
  --docker-registry-server-password <acr-password>
```

## 4. Configure secrets and ports

In the Azure Portal → App Service → Configuration → _Application settings_, add:

| Name             | Value (example) |
| ---------------- | --------------- |
| `AZURE_OPENAI_API_KEY` | `...` (Azure key) |
| `OPENAI_API_KEY` | *(optional)* only if you also call OpenAI directly |
| `AZURE_OPENAI_ENDPOINT` | `https://<resource>.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | `<your-deployment-name>` |
| `S2_API_KEY`     | `...`           |
| `COHERE_API_KEY` | `...`           |
| `PORT`           | `8000`          |
| `WEBSITES_PORT`  | `8000`          |

Save the settings so the container restarts with the new environment.

Notes:
- `AZURE_OPENAI_DEPLOYMENT` is the **deployment name** you created in Azure OpenAI Studio (not the model name).
- If you keep any `google:*` models in config, you will also need `GOOGLE_API_KEY`.

## 5. Verify and use the endpoint

```bash
curl https://paperfinder-api.azurewebsites.net/health -I
```

If you see `204`, the service is ready. Your frontend can now call `https://paperfinder-api.azurewebsites.net/api/2/rounds` instead of the temporary ngrok URL.

## Updating the deployment

1. Build a new image — **always with `--platform linux/amd64`** so the
   image runs on App Service regardless of whether the build host is
   Intel or Apple Silicon:

   ```bash
   docker buildx build --platform linux/amd64 -t paperfinder-api .
   ```

2. Push with a new tag: `docker push <acr>.azurecr.io/paperfinder-api:v2`
3. Point the Web App at the new tag:

   ```bash
   az webapp config container set \
     --name paperfinder-api \
     --resource-group roofaiopenai \
     --docker-custom-image-name <acr>.azurecr.io/paperfinder-api:v2
   ```

4. Restart the app (`az webapp restart ...`) or let the config change trigger a restart automatically.

That is all—no code changes are required when swapping tags, so this workflow fits easily into CI/CD.

## Troubleshooting

### `exec format error` in the docker log on boot

Symptom (from `LogFiles/<date>_<host>_default_docker.log`):
```
exec /app/.venv/bin/uvicorn: exec format error
```

Cause: the image was built on a different CPU architecture than the App
Service host. Apple Silicon (M1/M2/M3) Macs default to producing
`linux/arm64` images; App Service runs `linux/amd64`. The binaries
inside the venv are then the wrong architecture and Linux refuses to
`exec` them.

Fix: rebuild with `docker buildx build --platform linux/amd64 ...` as
shown above, push under a new tag, and update the App Service container
config to that tag.

### Container exits with code 255 within 230s of startup

Pull the docker log file from
`https://<app>.scm.azurewebsites.net/api/vfs/LogFiles/<date>_<host>_default_docker.log`
and look at the **last 50 lines** before the exit. There will usually be
a Python traceback (or an `exec ... error` like above). If you don't see
either, the container may have been killed by the App Service warm-up
probe before getting far enough to log; bump
`WEBSITES_CONTAINER_START_TIME_LIMIT` to `600` and retry.
