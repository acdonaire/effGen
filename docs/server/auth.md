# effGen API Server — OIDC / JWT Authentication

The effGen API server validates **Bearer JWTs** on every non-public endpoint.
Tokens are issued by any standard OIDC provider (Auth0, Keycloak, Google,
Azure AD, etc.).

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `EFFGEN_OIDC_ISSUER` | *(required in prod)* | Token issuer URL, e.g. `https://your-tenant.auth0.com/` |
| `EFFGEN_OIDC_CLIENT_ID` | *(required in prod)* | Expected `aud` claim in JWTs |
| `EFFGEN_OIDC_JWKS_URI` | auto-discovered | JWKS endpoint; omit to use OIDC discovery |
| `EFFGEN_DEV_MODE` | `0` | Set to `1` to disable auth with a loud warning |
| `EFFGEN_METRICS_AUTH` | `0` | Set to `1` to require auth on `/metrics` |

---

## Public endpoints (no auth required)

- `/health`
- `/healthz`, `/ready`, `/livez`
- `/metrics` (unless `EFFGEN_METRICS_AUTH=1`)

---

## Dev mode

```bash
EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory
```

A **loud warning** is printed to stderr. **Never** use `EFFGEN_DEV_MODE=1` in
production. In dev mode all requests are treated as the `dev-user` principal
with the `admin` role.

---

## Configuring Auth0

1. Create an API in Auth0 with identifier `https://effgen.api`.
2. Set environment variables:

```bash
export EFFGEN_OIDC_ISSUER=https://your-tenant.auth0.com/
export EFFGEN_OIDC_CLIENT_ID=https://effgen.api
# JWKS auto-discovered from /.well-known/openid-configuration
```

3. Obtain a token via the M2M flow:

```bash
TOKEN=$(curl -s -X POST "https://your-tenant.auth0.com/oauth/token" \
  -H "Content-Type: application/json" \
  -d '{"client_id":"<M2M_CLIENT_ID>","client_secret":"<SECRET>",
       "audience":"https://effgen.api","grant_type":"client_credentials"}' \
  | jq -r '.access_token')

curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/whoami
```

---

## Configuring Keycloak

```bash
export EFFGEN_OIDC_ISSUER=https://keycloak.example.com/realms/myrealm
export EFFGEN_OIDC_CLIENT_ID=effgen-api
# Keycloak exposes JWKS at {issuer}/protocol/openid-connect/certs
export EFFGEN_OIDC_JWKS_URI=https://keycloak.example.com/realms/myrealm/protocol/openid-connect/certs
```

Add a `roles` claim mapper to the Keycloak client to embed role names in the
JWT. effGen reads the `roles` claim (or `scope`) from the token.

---

## JWT claim format

effGen reads:

```json
{
  "sub": "user-id",
  "iss": "https://your-issuer/",
  "aud": "your-client-id",
  "exp": 1234567890,
  "roles": ["researcher", "admin"],
  "email": "user@example.com"
}
```

If `roles` is absent, the `scope` claim (space-separated) is used instead.

---

## Python API

```python
from effgen.server.auth import verify_jwt, TokenPayload

payload: TokenPayload = verify_jwt(
    raw_token,
    issuer="https://your-issuer/",
    client_id="your-client-id",
    jwks_uri="https://your-issuer/.well-known/jwks.json",
)
print(payload.sub, payload.roles)
```

### AuthMiddleware

```python
from effgen.server.auth import AuthMiddleware

app = FastAPI()
app.add_middleware(
    AuthMiddleware,
    issuer="https://your-issuer/",
    client_id="your-client-id",
    # jwks_uri auto-discovered if omitted
)
```

After the middleware runs, `request.state.user` is a `TokenPayload`.

---

## Security notes

- Tokens are validated against the JWKS public keys fetched from the IdP.
- The JWKS is cached for 1 hour to reduce latency.
- Expired tokens are always rejected.
- Audience and issuer mismatches are always rejected.
- The `Authorization` header value is never written to logs or audit records.
