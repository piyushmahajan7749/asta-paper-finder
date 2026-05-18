# Deploying PaperFinder to Azure App Service (Container)

This guide describes how to ship the FastAPI server (`/agents/mabool/api`) to Azure using a container image. You will end up with a stable `https://<app-name>.azurewebsites.net` URL that your frontend can call.

## 0. Prerequisites

- Azure CLI logged in (`az login`).
- An Azure resource group (use `az group create` if you do not have one).
- Docker and access to the repo root (where the `Dockerfile` lives).
- All required API keys handy (`OPENAI_API_KEY` for Azure OpenAI, `S2_API_KEY`, `COHERE_API_KEY`). (`GOOGLE_API_KEY` is only needed if you keep `google:*` models enabled.)

## 1. Build and test the image locally

```bash
cd /path/to/asta-paper-finder
docker build -t paperfinder-api .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=... \
  -e AZURE_OPENAI_ENDPOINT=... \
  -e AZURE_OPENAI_API_VERSION=... \
  -e AZURE_OPENAI_DEPLOYMENT=... \
  -e S2_API_KEY=... \
  -e COHERE_API_KEY=... \
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

1. Build a new image: `docker build -t paperfinder-api .`
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
