# DDPM FastAPI Web App: Docker build + run


Your existing files stay in place (generate.py, config.yaml, model/, utils/, etc.)

## 1) Build image (from project root)
```bash
docker build -t ddpm-web .
```

## 2) Run container 




### with gpu (mount checkpoint + publish port)


docker run --rm -it --name ddpm-web --gpus all -p 8081:8000 ddpm-web


### without gpu (mount checkpoint + publish port)

docker run --rm -it --name ddpm-web -p 8081:8000 ddpm-web

## 3) Open the UI
- http://localhost:8081/

## Notes
- If you don't have GPU support set up, remove `--gpus all`. docker run --rm -it --name ddpm-web -p 8081:8000 ddpm-web
- Health check: http://localhost:8081/health
