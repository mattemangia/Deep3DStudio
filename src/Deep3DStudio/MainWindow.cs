using System;
using Gtk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;

namespace Deep3DStudio
{
    public class MainWindow : Window
    {
        private ThreeDView _viewport;
        private Label _statusLabel;

        public MainWindow() : base(WindowType.Toplevel)
        {
            this.Title = "Deep3D Studio - Dust3r & NeRF";
            this.SetDefaultSize(1200, 800);
            this.DeleteEvent += (o, args) => Application.Quit();

            var mainVBox = new Box(Orientation.Vertical, 0);
            this.Add(mainVBox);

            // Toolbar
            var toolbar = CreateToolbar();
            mainVBox.PackStart(toolbar, false, false, 0);

            // Viewport and Side Panel (using Paned for resizing)
            var hPaned = new Paned(Orientation.Horizontal);
            mainVBox.PackStart(hPaned, true, true, 0);

            // 3D Viewport
            _viewport = new ThreeDView();
            hPaned.Pack1(_viewport, true, false);

            // Side Panel for Mesh Tools and Image List
            var sidePanel = CreateSidePanel();
            hPaned.Pack2(sidePanel, false, false);
            hPaned.Position = 900;

            // Status Bar
            _statusLabel = new Label("Ready");
            _statusLabel.Halign = Align.Start;
            var statusBox = new Box(Orientation.Horizontal, 5);
            statusBox.PackStart(_statusLabel, true, true, 5);
            mainVBox.PackStart(statusBox, false, false, 2);

            this.ShowAll();
        }

        private Widget CreateToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;

            // Add Image Button
            var addImgBtn = new ToolButton(IconGenerator.GenerateAddIcon(24), "Add Images");
            addImgBtn.TooltipText = "Add multiple images for processing";
            toolbar.Insert(addImgBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Run Dust3r Button
            var runBtn = new ToolButton(IconGenerator.GenerateRunIcon(24), "Run Dust3r");
            runBtn.TooltipText = "Run Dust3r Inference";
            toolbar.Insert(runBtn, -1);

            return toolbar;
        }

        private Widget CreateSidePanel()
        {
            var vbox = new Box(Orientation.Vertical, 5);
            vbox.Margin = 10;

            var label = new Label("Mesh Tools");
            label.Attributes = new Pango.AttrList();
            label.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(label, false, false, 5);

            // Tools with 3D handles
            var cropBtn = new Button("Crop Mesh");
            cropBtn.Clicked += (s, e) => {
                 _viewport.ToggleCropBox(true); // Simplified toggle
            };
            vbox.PackStart(cropBtn, false, false, 2);

            var editBtn = new Button("Edit Vertices");
            vbox.PackStart(editBtn, false, false, 2);

            var optimizeBtn = new Button("Optimize Mesh");
            vbox.PackStart(optimizeBtn, false, false, 2);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 10);

            var imgLabel = new Label("Input Images");
            imgLabel.Attributes = new Pango.AttrList();
            imgLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(imgLabel, false, false, 5);

            // Placeholder for image list
            var scrolled = new ScrolledWindow();
            scrolled.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);
            var treeView = new TreeView();
            scrolled.Add(treeView);
            vbox.PackStart(scrolled, true, true, 0);

            return vbox;
        }
    }
}
