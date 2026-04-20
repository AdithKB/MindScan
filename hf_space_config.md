# Deploying MindScan to Hugging Face Spaces

You can deploy this project to Hugging Face Spaces as a **Docker Space**.

### 1. Create a New Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- **Space Name**: e.g., `mindscan`
- **SDK**: Select **Docker**
- **Template**: Choose **Blank** (default)

### 2. Upload Files
You need to push your code to the Hugging Face Space repository. You can do this via Git.

```bash
# 1. Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>

# 2. Force push the code
git push hf main --force
```

### 3. Handle Large Model Files (Crucial)
The 3.2 GB of transformer models and classical `.pkl` files are **not** in the Git repository. You have two options for deployment:

#### Option A: Upload to HF LFS (Local mode)
Hugging Face Spaces supports large files via Git LFS.
1. Install [Git LFS](https://git-lfs.github.com/).
2. Place your `models/` folder into the root.
3. Configure LFS: `git lfs track "models/**"`
4. Push to Hugging Face.

#### Option B: Use Proxy Mode
If you already have a backend running elsewhere (like the provided `https://esvanth-mindscan.hf.space`), the `app.py` is already configured to automatically detect the absence of local models and forward requests to that URL.

### 4. Configuration
- The `Dockerfile` is already provided in the root.
- The `app.py` automatically uses the correct port (`7860`) for Hugging Face.
- The `.hfignore` file ensures that unnecessary files (like raw datasets and notebooks) are not uploaded to the deployment container.
