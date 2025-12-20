# Git Commands to Push Data Folder

## Step-by-Step Commands

### 1. Navigate to the KeyFrameDetection directory
```powershell
cd KeyFrameDetection
```

### 2. Initialize git repository (if not already initialized)
```powershell
git init
```

### 3. Check current status
```powershell
git status
```

### 4. Add the data folder to staging

**Since you're already in KeyFrameDetection directory, use relative paths:**

To add just the video folder:
```powershell
git add data/Highway_Traffic/video
```

To add the entire Highway_Traffic folder:
```powershell
git add data/Highway_Traffic/
```

To add the entire data folder:
```powershell
git add data/
```

### 5. Commit the changes
```powershell
git commit -m "Add data folder"
```

### 6. Add remote repository (if not already added)
```powershell
git remote add origin <your-repository-url>
```

### 7. Check remote repository
```powershell
git remote -v
```

### 8. Push to remote repository
```powershell
git push -u origin main
```

**OR if your branch is named 'master':**
```powershell
git push -u origin master
```

---

## Complete Command Sequence (Copy & Paste)

```powershell
# Navigate to directory
cd KeyFrameDetection

# Initialize git (if needed)
git init

# Add data folder
git add data/

# Commit
git commit -m "Add data folder"

# Add remote (replace with your actual repository URL)
git remote add origin <your-repository-url>

# Push to remote
git push -u origin main
```

---

## Notes

⚠️ **Warning**: The data folder contains many video files (.avi) and image files (.jpg) which are large. This may take a long time to upload.

If you want to exclude large files in the future, you can uncomment lines 21-25 in `.gitignore`:
```
*.mp4
*.avi
*.mov
data/videos/*
data/frames/*
```

