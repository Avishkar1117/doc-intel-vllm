# infra/ — Azure provisioning, documented as `az` CLI (Phase 7)

No Bicep/Terraform here on purpose (see `FUTURE.md`) — these are the exact commands that
built the live deployment, in order, so anyone (including future-you) can reproduce it or
understand what exists without guessing. All resource names below are the real ones in use.

Prerequisites: `az login`, an active subscription, the `containerapp` and
`application-insights` CLI extensions (`az extension add --name containerapp --upgrade` /
`az extension add -n application-insights`).

## Resources created (all in `rg-docintel`, Germany West Central)

| Resource | Name | Purpose |
|---|---|---|
| Resource group | `rg-docintel` | everything below lives here |
| Container Registry | `acrdocintel90b9a600` | holds the app image |
| Storage account | `stdocintel90b9a600` | Blob container `receipts`, 24h lifecycle delete |
| Log Analytics workspace | `law-docintel` | backs App Insights |
| Application Insights | `appi-docintel` | telemetry (workspace-based) |
| Key Vault | `kvdocintel90b9a600` | RBAC-mode; holds the extract API key + Modal token |
| Container Apps environment | `env-docintel` | **WorkloadProfiles mode**, not Express — see below |
| Container App | `app-docintel` | the running FastAPI app |

The `90b9a600` suffix exists only because the generic names collided with other Azure
customers globally (ACR/Storage/Key Vault names are globally unique) — it has no other
meaning.

## 1. Resource group

```bash
az group create --name rg-docintel --location germanywestcentral
```

New subscriptions don't have most resource providers registered yet; if creation fails with
`MissingSubscriptionRegistration`, register what's needed first:

```bash
for ns in Microsoft.ContainerRegistry Microsoft.Storage Microsoft.OperationalInsights \
          Microsoft.Insights Microsoft.KeyVault Microsoft.App Microsoft.ManagedIdentity \
          Microsoft.OperationsManagement; do
  az provider register --namespace $ns
done
# then poll: az provider show -n <namespace> --query registrationState
```

## 2. ACR, Storage, Blob container + 24h lifecycle policy

```bash
az acr create --resource-group rg-docintel --name acrdocintel90b9a600 --sku Basic \
  --location germanywestcentral

az storage account create --name stdocintel90b9a600 --resource-group rg-docintel \
  --location germanywestcentral --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2

az storage container create --name receipts --account-name stdocintel90b9a600 --auth-mode login
```

Lifecycle policy (delete after 24h — `daysAfterModificationGreaterThan: 1`; note this runs on
a daily sweep, so actual deletion can lag up to ~24-48h past the literal 24h mark):

```bash
cat > lifecycle_policy.json << 'EOF'
{
  "rules": [
    {
      "enabled": true,
      "name": "delete-after-24h",
      "type": "Lifecycle",
      "definition": {
        "actions": { "baseBlob": { "delete": { "daysAfterModificationGreaterThan": 1 } } },
        "filters": { "blobTypes": ["blockBlob"], "prefixMatch": ["receipts/"] }
      }
    }
  ]
}
EOF
az storage account management-policy create --account-name stdocintel90b9a600 \
  --resource-group rg-docintel --policy @lifecycle_policy.json
```

## 3. Log Analytics + Application Insights

```bash
az monitor log-analytics workspace create --resource-group rg-docintel \
  --workspace-name law-docintel --location germanywestcentral

WS_ID=$(az monitor log-analytics workspace show --resource-group rg-docintel \
  --workspace-name law-docintel --query id -o tsv)
az monitor app-insights component create --app appi-docintel --location germanywestcentral \
  --resource-group rg-docintel --kind web --application-type web --workspace "$WS_ID"
```

**Windows/Git Bash gotcha:** any argument that looks like a POSIX path (starts with `/`,
e.g. `$WS_ID`) gets silently rewritten to a Windows path by Git Bash's MSYS layer unless you
set `MSYS_NO_PATHCONV=1` first. Every command below that passes a `/subscriptions/...`
resource ID needs this.

## 4. Key Vault (RBAC mode)

```bash
az keyvault create --name kvdocintel90b9a600 --resource-group rg-docintel \
  --location germanywestcentral --enable-rbac-authorization true
```

**Gotcha:** creating a Key Vault with RBAC authorization does **not** grant the creator any
data-plane access to it — you must explicitly grant yourself a role before you can read or
write secrets, same as any other principal:

```bash
export MSYS_NO_PATHCONV=1
MY_ID=$(az ad signed-in-user show --query id -o tsv)
KV_ID=$(az keyvault show -n kvdocintel90b9a600 --query id -o tsv)
az role assignment create --assignee "$MY_ID" --role "Key Vault Secrets Officer" --scope "$KV_ID"
```

## 5. Container Apps environment — **WorkloadProfiles, not Express**

```bash
WS_ID=$(az monitor log-analytics workspace show -g rg-docintel -n law-docintel \
  --query customerId -o tsv)
WS_KEY=$(az monitor log-analytics workspace get-shared-keys -g rg-docintel -n law-docintel \
  --query primarySharedKey -o tsv)
az containerapp env create -n env-docintel -g rg-docintel --location germanywestcentral \
  --environment-mode WorkloadProfiles --logs-workspace-id "$WS_ID" --logs-workspace-key "$WS_KEY"
```

**Real gotcha, cost real time:** `az containerapp env create` without `--environment-mode`
silently defaults to Azure's newer **Express** tier, a lighter environment type that does
**not** support system-assigned managed identity, Key Vault secret references, or
OpenTelemetry — three things this deployment needs. If you ever see
`ExpressEnvironmentFeatureNotSupported`, this is why. Always pass
`--environment-mode WorkloadProfiles` explicitly. `workloadProfiles` should show only the
free, scale-to-zero `Consumption` profile — verify with
`az containerapp env show -n env-docintel -g rg-docintel --query properties.workloadProfiles`.

## 6. Build and push the image — **not `az acr build`**

This subscription is Azure for Students, where **ACR Tasks are blocked outright**
(`TasksOperationsNotAllowed`) as an anti-abuse measure on trial-tier subscriptions. Build
locally with Docker instead:

```bash
az acr login --name acrdocintel90b9a600
docker build -t acrdocintel90b9a600.azurecr.io/docintel-api:<tag> -f docker/Dockerfile.api .
docker push acrdocintel90b9a600.azurecr.io/docintel-api:<tag>
```

(The GitHub Actions `deploy.yml` workflow does the same thing, tagged by commit SHA.)

## 7. Container App — bootstrap, then swap to the real image

The identity doesn't exist until the app exists, so you can't grant it ACR pull access
before creating it, and you can't reference a private image before it has access — hence a
two-step bootstrap with a public placeholder image first:

```bash
az containerapp create -n app-docintel -g rg-docintel --environment env-docintel \
  --image mcr.microsoft.com/k8se/quickstart:latest --target-port 8000 --ingress external \
  --system-assigned

export MSYS_NO_PATHCONV=1
PRINCIPAL_ID=$(az containerapp show -n app-docintel -g rg-docintel \
  --query identity.principalId -o tsv)
ACR_ID=$(az acr show -n acrdocintel90b9a600 --query id -o tsv)
az role assignment create --assignee "$PRINCIPAL_ID" --role AcrPull --scope "$ACR_ID"

az containerapp registry set -n app-docintel -g rg-docintel --identity system \
  --server acrdocintel90b9a600.azurecr.io
az containerapp update -n app-docintel -g rg-docintel \
  --image acrdocintel90b9a600.azurecr.io/docintel-api:<tag>
```

Role assignments can take up to a couple of minutes to propagate — if the pull/secret-fetch
fails immediately after granting a role, wait and retry before assuming something's wrong.

## 8. Secrets — Key Vault + Managed Identity, no plaintext anywhere

Grant the app's identity read access, store each secret once, reference it by URL, expose it
as an env var:

```bash
export MSYS_NO_PATHCONV=1
PRINCIPAL_ID=$(az containerapp show -n app-docintel -g rg-docintel \
  --query identity.principalId -o tsv)
KV_ID=$(az keyvault show -n kvdocintel90b9a600 --query id -o tsv)
az role assignment create --assignee "$PRINCIPAL_ID" --role "Key Vault Secrets User" --scope "$KV_ID"

az keyvault secret set --vault-name kvdocintel90b9a600 --name extract-api-key \
  --value "$(openssl rand -hex 32)"
az containerapp secret set -n app-docintel -g rg-docintel \
  --secrets extract-api-key=keyvaultref:https://kvdocintel90b9a600.vault.azure.net/secrets/extract-api-key,identityref:system
az containerapp update -n app-docintel -g rg-docintel \
  --set-env-vars DOCINTEL_EXTRACT_API_KEY=secretref:extract-api-key
```

The Modal vLLM proxy auth token (`modal-vllm-api-key`) is stored the same way — see §10.

Blob access uses the same identity, a different role:

```bash
export MSYS_NO_PATHCONV=1
PRINCIPAL_ID=$(az containerapp show -n app-docintel -g rg-docintel --query identity.principalId -o tsv)
ST_ID=$(az storage account show -n stdocintel90b9a600 --query id -o tsv)
az role assignment create --assignee "$PRINCIPAL_ID" --role "Storage Blob Data Contributor" --scope "$ST_ID"
az containerapp update -n app-docintel -g rg-docintel \
  --set-env-vars DOCINTEL_STORAGE_ACCOUNT_NAME=stdocintel90b9a600 DOCINTEL_BLOB_CONTAINER=receipts
```

App Insights isn't a credential (it can only ingest telemetry, not grant access to
anything), so it's just a plain env var, not Key-Vaulted:

```bash
az containerapp update -n app-docintel -g rg-docintel \
  --set-env-vars APPLICATIONINSIGHTS_CONNECTION_STRING="<connection string>"
```

## 9. Verifying the deployment

```bash
curl https://app-docintel.purplebay-f7aec3f1.germanywestcentral.azurecontainerapps.io/health
az containerapp revision list -n app-docintel -g rg-docintel \
  --query "[].{name:name, healthState:properties.healthState}" -o table
az containerapp logs show -n app-docintel -g rg-docintel --tail 60
az monitor app-insights query --app appi-docintel -g rg-docintel \
  --analytics-query "requests | order by timestamp desc | take 10" --offset 1h
```

## 10. Modal proxy auth token

The vLLM server (`modal_app.py`) runs `unauthenticated=False` — every caller needs a Modal
Proxy Auth Token:

```bash
uv run modal workspace proxy-tokens create
```

Combined bearer format is `<token-id>.<token-secret>`, sent as
`Authorization: Bearer <token-id>.<token-secret>` — `extraction/client.py`'s `HTTPClient`
already sends exactly this when `DOCINTEL_VLLM_API_KEY` is set. Store and wire it the same
way as `extract-api-key` in §8, as `modal-vllm-api-key`, exposed as `DOCINTEL_VLLM_API_KEY`.

## 11. GitHub Actions OIDC (for `deploy.yml`)

One-time identity setup so CI can deploy without ever holding an Azure secret:

```bash
az ad app create --display-name "docintel-github-deploy" --query "{appId:appId, id:id}"
az ad sp create --id <appId>

cat > fed-cred.json << 'EOF'
{
  "name": "github-main-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<OWNER>@<OWNER_ID>/<REPO>@<REPO_ID>:ref:refs/heads/master",
  "description": "GitHub Actions deploy workflow",
  "audiences": ["api://AzureADTokenExchange"]
}
EOF
az ad app federated-credential create --id <app-object-id> --parameters fed-cred.json
```

**Real gotcha:** repos created after GitHub's mid-2026 cutover use an **immutable subject
format** that embeds numeric owner/repo IDs
(`repo:OWNER@OWNER_ID/REPO@REPO_ID:ref:refs/heads/BRANCH`), not the plain
`repo:OWNER/REPO:ref:refs/heads/BRANCH` every tutorial shows. If login fails with
`AADSTS700213: No matching federated identity record found`, the error message itself
includes the exact subject string GitHub actually presented — copy that verbatim into
`az ad app federated-credential update --id <app-object-id> --federated-credential-id
github-main-branch --parameters '{"subject":"<exact string from the error>"}'`.

Grant the identity what it needs, then add the three GitHub secrets:

```bash
export MSYS_NO_PATHCONV=1
SP_ID=$(az ad sp show --id <appId> --query id -o tsv)
ACR_ID=$(az acr show -n acrdocintel90b9a600 --query id -o tsv)
RG_ID=$(az group show -n rg-docintel --query id -o tsv)
az role assignment create --assignee "$SP_ID" --role "AcrPush" --scope "$ACR_ID"
az role assignment create --assignee "$SP_ID" --role "Container Apps Contributor" --scope "$RG_ID"

gh secret set AZURE_CLIENT_ID --body "<appId>"
gh secret set AZURE_TENANT_ID --body "<tenant-id>"
gh secret set AZURE_SUBSCRIPTION_ID --body "<subscription-id>"
```

## 12. Phase 9: the public demo page

Two more private containers in the same storage account (outside the lifecycle policy's
`receipts/` prefix, so nothing here is auto-deleted), one for the ten cached samples and one
for the spend ledger's state blob:

```bash
az storage container create --name demo-samples --account-name stdocintel90b9a600 --auth-mode login
az storage container create --name ledger --account-name stdocintel90b9a600 --auth-mode login
```

**Gotcha, second time:** uploading with `--auth-mode login` fails with a permissions error
even for the account owner. Owner is a control-plane role; blob data needs its own grant
(same class as the Key Vault gotcha in §4):

```bash
export MSYS_NO_PATHCONV=1
MY_ID=$(az ad signed-in-user show --query id -o tsv)
ST_ID=$(az storage account show -n stdocintel90b9a600 --query id -o tsv)
az role assignment create --assignee "$MY_ID" --role "Storage Blob Data Contributor" --scope "$ST_ID"
# role propagation took about a minute; retry the upload until it works
```

Build the sample bundle locally (needs the SROIE parquet under `data/`), then upload it. The
images are not in git; only `samples.json` and the images in this container serve the page:

```bash
uv run python benchmarks/build_demo_samples.py
az storage blob upload-batch --account-name stdocintel90b9a600 --destination demo-samples \
  --source demo_samples --auth-mode login --overwrite
```

The demo's rate limiter and the spend ledger assume exactly one app replica:

```bash
az containerapp update -n app-docintel -g rg-docintel --min-replicas 0 --max-replicas 1
```

The demo GPU is a separate Modal app in its own environment (`modal_demo.py`; one L4,
five-minute idle window, `max_containers=1`). It mounts the weights Volume from `main`, so
nothing is downloaded again:

```bash
uv run modal environment create demo
uv run modal deploy -e demo modal_demo.py     # prints the *.modal.direct URL
```

Point the Container App at it. The Modal proxy token is the same workspace token already
stored as the `modal-vllm-api-key` secret, so no new secret is needed. The scaledown value
must equal `scaledown_window` in `modal_demo.py` (it prices the idle tail in the ledger):

```bash
az containerapp update -n app-docintel -g rg-docintel --set-env-vars \
  "DOCINTEL_DEMO_VLLM_BASE_URL=<the modal.direct URL>" \
  "DOCINTEL_DEMO_VLLM_API_KEY=secretref:modal-vllm-api-key" \
  "DOCINTEL_DEMO_SCALEDOWN_WINDOW_S=300"
```

Check the ledger and the page:

```bash
az storage blob download --account-name stdocintel90b9a600 --container-name ledger \
  --name state.json --file - --auth-mode login
curl https://<app fqdn>/demo/status
```

Keep the demo's usage history for the months the site stays up. The workspace default is 30
days; 180 days of this project's volume (under 1 MB a month) costs a fraction of a cent, since
only storage beyond the free 31 days is billed:

```bash
az monitor log-analytics workspace update -g rg-docintel -n law-docintel --retention-time 180
```

**What "Wake the GPU" costs:** the first request to a sleeping Modal server is rejected with a
503 but starts a container; measured time to healthy was 313 s. The ledger charges a warm-up
a cold start plus one idle window (about $0.17 with the 1.1 margin), so a $5 month is roughly
30 cold visits before live extraction switches off and only the cached samples remain.

## Teardown

This deployment exists to prove the engineering happened, not to run forever (§4 of
`PROJECT_BRIEF.md`: reproducible evidence belongs in the repo, not in a live endpoint). Once
Phase 9's demo recording is done, tear the whole thing down in one command — Azure Container
Registry is the one resource here with a real fixed daily cost regardless of usage,
everything else is usage-based and near-$0 when idle:

```bash
az group delete --name rg-docintel --yes
```

This deletes every resource this document created, including the Document Intelligence
resource (`docintel-di`, F0) and both new containers. The Modal side is separate and not
covered by it; stop both apps (`main` normally already scaled to zero):

```bash
uv run modal app stop vllm-doc-intelligence-demo -e demo -y
uv run modal app stop vllm-doc-intelligence -y
```

The GitHub Actions identity
(`docintel-github-deploy`) and its role assignments are not inside the resource group and
need a separate cleanup if you want them gone too:

```bash
az ad app delete --id <appId>
```
