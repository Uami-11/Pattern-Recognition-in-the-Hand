## To run the web thing

```
python export_model.py

python -m http.server 8000
```

## Deploy the web demo to Vercel

`vercel.json` builds a static `dist/` folder containing `web/` and `sounds/`.
No environment variables or server-side runtime are needed.

From the repository root:

```bash
npx vercel          # preview deployment
npx vercel --prod   # production deployment
```

The first run links the project to your Vercel account. Vercel serves the
deployed site over HTTPS, so camera access works without extra setup.
