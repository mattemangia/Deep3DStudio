<p align="center">
  <img src="src/Deep3DStudio/logo.png" alt="Deep3D Studio Logo" width="200"/>
</p>

<h1 align="center">Deep3D Studio</h1>

<p align="center">
  <strong>AI-Powered 3D Reconstruction</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#building-from-source">Building</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

---

## Overview

Deep3D Studio is a comprehensive AI-powered 3D reconstruction application that combines state-of-the-art deep learning models with traditional computer vision techniques. It enables users to create high-quality 3D models from images using various AI pipelines, with support for mesh processing, rigging, texturing, and more.

The application is developed as part of research at the **Università degli Studi di Urbino Carlo Bo**.

## Features

### AI-Powered 3D Reconstruction

| Model | Description |
|-------|-------------|
| **TripoSR** | Fast single-image to 3D generation |
| **TripoSF** | High-resolution 3D mesh generation |
| **LGM** | Large Multi-View Gaussian Model for high-quality reconstruction |
| **Wonder3D** | Multi-view image generation and reconstruction |
| **Dust3r** | Multi-view reconstruction with AI enhancement |

### Mesh Processing & Refinement

- **DeepMeshPrior**: AI-based mesh optimization
- **TripoSF (SparseFormer)**: Mesh refinement
- **GaussianSDF**: Signed distance field refinement
- **NeRF Models**: Neural Radiance Field refinement

### Mesh Operations

- Decimation (polygon reduction)
- Smoothing and optimization
- Connectivity-based splitting
- Normal flipping and merging
- ICP (Iterative Closest Point) alignment
- Cleanup tools
- Triangle-level editing with pen tool

### Rigging & Skeleton

- Manual rigging tool
- Skeleton creation and management
- Humanoid template pre-sets
- **UniRig**: Automatic rigging system
- Rigged mesh export

### Meshing Algorithms

- Marching Cubes
- Poisson Reconstruction
- SurfaceNets
- GreedyMeshing
- Blocky meshing

### Texturing

- Texture baking
- Texture projection

### Point Cloud Operations

- Import/export point clouds
- Merging and alignment
- Voxelization
- Filtering

### Advanced Workflows

- Full pipeline: Images → Dust3r → NeRF → Mesh
- Dust3r → DeepMeshPrior workflow
- Dust3r → NeRF → DeepMeshPrior workflow
- Point cloud merge & refine
- SfM → AI refinement

### Import/Export Formats

| Type | Formats |
|------|---------|
| **Mesh** | OBJ, FBX, PLY, GLTF |
| **Point Clouds** | PLY, XYZ, LAS |
| **Other** | Rigged meshes, Textured meshes, Depth maps |

### Viewport & Visualization

- OpenGL-based 3D visualization
- Mesh/point cloud display modes
- Wireframe rendering
- RGB and depth color modes
- Grid, axes, and camera frustum visualization
- Transform gizmo (Move/Rotate/Scale)
- Selection tools
- Info overlay with FPS counter

## Installation

### Pre-built Releases

Download the latest release for your platform from the [Releases](https://github.com/mattemangia/Deep3DStudio/releases) page.

| Platform | Download |
|----------|----------|
| Windows (x64) | `Deep3DStudio-win-x64.zip` |
| Linux (x64) | `Deep3DStudio-linux-x64.tar.gz` |
| macOS (ARM64) | `Deep3DStudio-osx-arm64.tar.gz` |
| macOS (x64) | `Deep3DStudio-osx-x64.tar.gz` |

### System Requirements

- **OS**: Windows 10+, Linux (Ubuntu 20.04+), macOS 11+
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support recommended (for AI features)
- **Storage**: 10GB+ for application and AI models

## Usage

### Getting Started

1. **Launch the application**
   - Windows: Run `Deep3DStudio.Cross.exe`
   - Linux/macOS (GTK): Run `Deep3DStudio`
   - Linux/macOS (Cross): Run `Deep3DStudio.Cross`

2. **Load images**
   - Use `File → Import Images` to load your source images
   - Supported formats: PNG, JPG, JPEG, BMP

3. **Choose reconstruction method**
   - Select your preferred AI model from the dropdown in the left panel
   - Options: Dust3r, TripoSR, LGM, Wonder3D

4. **Run reconstruction**
   - Click the "Reconstruct" button or use `Ctrl+R`
   - Monitor progress in the status bar

5. **Refine and edit**
   - Use mesh operations (decimation, smoothing, etc.)
   - Apply AI refinement if needed
   - Edit manually with the pen tool

6. **Export**
   - Use `File → Export` to save your 3D model
   - Choose from OBJ, FBX, PLY, or GLTF formats

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+I` | Import images |
| `Ctrl+E` | Export mesh |
| `Ctrl+R` | Run reconstruction |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `F` | Focus on selection |
| `G` | Toggle grid |
| `W` | Toggle wireframe |
| `Delete` | Delete selection |

### Viewport Navigation

| Action | Control |
|--------|---------|
| Rotate view | Left mouse + drag |
| Pan view | Middle mouse + drag |
| Zoom | Scroll wheel |
| Select object | Left click |
| Multi-select | Ctrl + Left click |

### Project Files

Deep3D Studio uses `.d3d` project files to save:
- All loaded meshes and point clouds
- Scene hierarchy
- Camera positions
- Settings and preferences

## Building from Source

### Prerequisites

- **.NET 8.0 SDK** or higher
- **Python 3.10+** (for AI models)
- **Git**

#### Platform-Specific Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libgtk-3-dev libglib2.0-dev
```

**macOS:**
```bash
brew install gtk+3
```

**Windows:**
No additional dependencies required for the cross-platform (ImGui) version.

### Build Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/mattemangia/Deep3DStudio.git
   cd Deep3DStudio
   ```

2. **Restore dependencies**
   ```bash
   dotnet restore Deep3DStudio.sln
   ```

3. **Build the solution**
   ```bash
   # Build all projects
   dotnet build Deep3DStudio.sln --configuration Release

   # Or build specific project:
   # GTK version (Linux/macOS)
   dotnet build src/Deep3DStudio/Deep3DStudio.csproj --configuration Release

   # Cross-platform version (Windows/Linux/macOS)
   dotnet build src/Deep3DStudio.Cross/Deep3DStudio.Cross.csproj --configuration Release
   ```

4. **Run the application**
   ```bash
   # GTK version
   dotnet run --project src/Deep3DStudio/Deep3DStudio.csproj

   # Cross-platform version
   dotnet run --project src/Deep3DStudio.Cross/Deep3DStudio.Cross.csproj
   ```

### Publishing for Distribution

```bash
# Windows
dotnet publish src/Deep3DStudio.Cross/Deep3DStudio.Cross.csproj \
  --configuration Release \
  --runtime win-x64 \
  --self-contained true \
  -p:PublishSingleFile=true

# Linux
dotnet publish src/Deep3DStudio.Cross/Deep3DStudio.Cross.csproj \
  --configuration Release \
  --runtime linux-x64 \
  --self-contained true \
  -p:PublishSingleFile=true

# macOS (ARM64)
dotnet publish src/Deep3DStudio.Cross/Deep3DStudio.Cross.csproj \
  --configuration Release \
  --runtime osx-arm64 \
  --self-contained true \
  -p:PublishSingleFile=true
```

### Setting Up Python Environment

For AI model support, run the deployment setup script:

```bash
python scripts/setup_deployment.py --platform [win_amd64|linux_x64|osx_arm64]
```

This will download:
- Standalone Python 3.10.11
- AI model repositories (Dust3r, TripoSR, TripoSF, LGM, Wonder3D, UniRig)
- Pre-trained model weights
- Required Python dependencies

## Project Structure

```
Deep3DStudio/
├── src/
│   ├── Deep3DStudio/              # GTK+ GUI (Linux/macOS native)
│   │   ├── Configuration/         # Settings management
│   │   ├── DeepMeshPrior/         # Deep learning mesh refinement
│   │   ├── Icons/                 # Application icons
│   │   ├── IO/                    # Import/export handlers
│   │   ├── Meshing/               # Meshing algorithms
│   │   ├── Model/                 # AI inference & geometry
│   │   ├── Python/                # Python runtime integration
│   │   ├── Scene/                 # Scene graph & objects
│   │   ├── Texturing/             # Texture processing
│   │   ├── UI/                    # Dialog windows & panels
│   │   └── Viewport/              # OpenGL rendering
│   └── Deep3DStudio.Cross/        # Cross-platform ImGui version
├── scripts/
│   ├── dist/                      # Deployment packages
│   └── setup_deployment.py        # Setup automation
├── .github/workflows/             # CI/CD automation
├── requirements.txt               # Python dependencies
└── Deep3DStudio.sln              # Visual Studio solution
```

## Contributing

We welcome contributions to Deep3D Studio! Here's how you can help:

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Guidelines

#### Code Style

- Follow C# coding conventions (Microsoft guidelines)
- Use meaningful variable and method names
- Add XML documentation for public APIs
- Keep methods focused and concise

#### Python Code

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Document functions with docstrings

#### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issues when applicable: `Fix #123: Description`

### Pull Request Process

1. **Ensure your code builds** without errors or warnings
2. **Test your changes** thoroughly
3. **Update documentation** if needed
4. **Create a Pull Request** with:
   - Clear description of changes
   - Screenshots for UI changes
   - Reference to related issues

### Reporting Issues

When reporting bugs, please include:
- Operating system and version
- Steps to reproduce the issue
- Expected vs actual behavior
- Screenshots or error logs if applicable

### Feature Requests

For feature requests:
- Check existing issues first
- Describe the use case clearly
- Explain why this would benefit users

## Technologies Used

### Core Framework
- **.NET 8.0** - Cross-platform runtime
- **C#** - Primary language
- **Python** - AI model inference

### GUI Frameworks
- **GTK# 3** - Native Linux/macOS GUI
- **ImGui.NET** - Cross-platform GUI (Windows/Linux/macOS)

### Graphics & Visualization
- **OpenTK** - OpenGL bindings
- **SkiaSharp** - 2D graphics

### AI & Machine Learning
- **PyTorch** - Deep learning framework
- **TorchSharp** - PyTorch .NET bindings
- **pythonnet** - Python interop

### Computer Vision
- **OpenCvSharp4** - OpenCV bindings

### Math & Numerics
- **MathNet.Numerics** - Mathematical computations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

**Author:** Matteo Mangiagalli
**Email:** m.mangiagalli@campus.uniurb.it
**Institution:** Università degli Studi di Urbino Carlo Bo
**Year:** 2026

### Acknowledgments

This project incorporates or builds upon the following open-source projects:

- [Dust3r](https://github.com/naver/dust3r) - Multi-view reconstruction
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) - Single-image 3D generation
- [TripoSF](https://github.com/VAST-AI-Research/TripoSF) - High-resolution 3D mesh generation
- [LGM](https://github.com/3DTopia/LGM) - Large Multi-View Gaussian Model
- [Wonder3D](https://github.com/xxlong0/Wonder3D) - Multi-view generation
- [UniRig](https://github.com/VAST-AI-Research/UniRig) - Automatic rigging

---

<p align="center">
  Made with dedication at Università degli Studi di Urbino Carlo Bo
</p>
