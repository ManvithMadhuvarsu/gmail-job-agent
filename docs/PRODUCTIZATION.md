# Productizing MailAI

This repo can be sold first as a local/self-hosted product before becoming a hosted SaaS.

## Recommended Offers

- MailAI Local: annual license for one user running on their own machine.
- MailAI Pro: Docker/self-hosted install with Calendar backfill and setup support.
- MailAI Concierge: done-for-you deployment, OAuth setup, Ollama/Groq configuration, and support.

## Why Local First

Gmail access is sensitive. A hosted SaaS needs Google OAuth verification, strong privacy pages,
encrypted token handling, deletion workflows, and possibly a security assessment. A local/self-hosted
offer lets customers use their own Google OAuth and LLM keys while you prove demand.

## License Setup

Create a seller signing keypair:

```bash
python scripts/generate_license.py init-keys
```

Keep `data/license_private.key` secret. Do not commit it.

Copy the printed public key into paid builds:

```env
MAILAI_LICENSE_REQUIRED=true
MAILAI_LICENSE_PUBLIC_KEY=<printed public key>
```

Issue a customer license:

```powershell
python scripts/generate_license.py issue `
  --customer "Customer Name" `
  --email buyer@example.com `
  --tier pro `
  --expires-at 2027-05-29
```

The customer can paste the generated token at `/license` or set:

```env
MAILAI_LICENSE_KEY=mailai_v1...
```

## Launch Checklist

- Create a landing page and pricing page. The FastAPI root page now provides a product-grade start.
- Publish privacy policy, terms, and data deletion instructions before accepting public users.
- Keep auto-send out of scope. Drafts should remain review-only.
- Package local setup with Docker Compose and a guided `.env` template.
- For hosted SaaS, complete Google OAuth verification before onboarding external users.
