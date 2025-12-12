using System;

using Gtk;

using Deep3DStudio.Scene;



namespace Deep3DStudio.UI

{

    /// <summary>

    /// Dialog for configuring mesh cleanup operations

    /// </summary>

    public class MeshCleanupDialog : Dialog

    {

        // Checkboxes for each operation

        private CheckButton _removeDegenerateTris;

        private CheckButton _removeSlivers;

        private CheckButton _removeSmallComponents;

        private CheckButton _removeStatisticalOutliers;

        private CheckButton _fixNonManifold;

        private CheckButton _fillHoles;

        private CheckButton _fixNormals;

        private CheckButton _removeIsolatedVertices;



        // Parameters

        private SpinButton _minTriangleAreaSpin;

        private SpinButton _minAspectRatioSpin;

        private SpinButton _minComponentSizeSpin;

        private SpinButton _kNeighborsSpin;

        private SpinButton _stdRatioSpin;

        private SpinButton _maxHoleEdgesSpin;



        // Presets

        private ComboBoxText _presetCombo;



        public MeshCleanupOptions Options { get; private set; } = new MeshCleanupOptions();



        public MeshCleanupDialog(Window parent, int currentVertexCount, int currentTriangleCount)

            : base("Mesh Cleanup", parent, DialogFlags.Modal)

        {

            SetDefaultSize(500, 600);

            BuildUI(currentVertexCount, currentTriangleCount);

            LoadPreset("Default");

        }



        private void BuildUI(int vertexCount, int triangleCount)

        {

            var contentArea = ContentArea;

            contentArea.Margin = 10;

            contentArea.Spacing = 10;



            // Info label

            var infoLabel = new Label($"<b>Current mesh:</b> {vertexCount:N0} vertices, {triangleCount:N0} triangles");

            infoLabel.UseMarkup = true;

            infoLabel.Halign = Align.Start;

            contentArea.PackStart(infoLabel, false, false, 0);



            // Preset selection

            var presetBox = new Box(Orientation.Horizontal, 5);

            presetBox.PackStart(new Label("Preset:"), false, false, 0);

            _presetCombo = new ComboBoxText();

            _presetCombo.AppendText("Default");

            _presetCombo.AppendText("Conservative");

            _presetCombo.AppendText("Aggressive");

            _presetCombo.AppendText("Scan Cleanup");

            _presetCombo.AppendText("Custom");

            _presetCombo.Active = 0;

            _presetCombo.Changed += OnPresetChanged;

            presetBox.PackStart(_presetCombo, true, true, 0);

            contentArea.PackStart(presetBox, false, false, 0);



            contentArea.PackStart(new Separator(Orientation.Horizontal), false, false, 5);



            // Scrolled options

            var scrolled = new ScrolledWindow();

            scrolled.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);

            scrolled.ShadowType = ShadowType.In;



            var optionsBox = new Box(Orientation.Vertical, 5);

            optionsBox.Margin = 10;



            // Remove Degenerate Triangles

            var degenerateFrame = CreateOperationFrame("Remove Degenerate Triangles",

                "Removes zero-area triangles and triangles with duplicate vertices",

                out _removeDegenerateTris);

            var degenerateGrid = new Grid { ColumnSpacing = 10, RowSpacing = 5, Margin = 5 };

            degenerateGrid.Attach(new Label("Min Area:") { Halign = Align.Start }, 0, 0, 1, 1);

            _minTriangleAreaSpin = new SpinButton(0, 0.001, 0.0000001);

            _minTriangleAreaSpin.Value = 0.00000001;

            _minTriangleAreaSpin.Digits = 10;

            degenerateGrid.Attach(_minTriangleAreaSpin, 1, 0, 1, 1);

            ((Box)degenerateFrame.Child).PackStart(degenerateGrid, false, false, 0);

            optionsBox.PackStart(degenerateFrame, false, false, 0);



            // Remove Slivers

            var sliverFrame = CreateOperationFrame("Remove Sliver Triangles",

                "Removes long thin triangles that can cause rendering issues",

                out _removeSlivers);

            var sliverGrid = new Grid { ColumnSpacing = 10, RowSpacing = 5, Margin = 5 };

            sliverGrid.Attach(new Label("Min Aspect Ratio:") { Halign = Align.Start }, 0, 0, 1, 1);

            _minAspectRatioSpin = new SpinButton(0.001, 0.5, 0.01);

            _minAspectRatioSpin.Value = 0.01;

            _minAspectRatioSpin.Digits = 3;

            sliverGrid.Attach(_minAspectRatioSpin, 1, 0, 1, 1);

            ((Box)sliverFrame.Child).PackStart(sliverGrid, false, false, 0);

            optionsBox.PackStart(sliverFrame, false, false, 0);



            // Remove Small Components

            var smallFrame = CreateOperationFrame("Remove Small Components",

                "Removes disconnected mesh parts below a vertex threshold",

                out _removeSmallComponents);

            var smallGrid = new Grid { ColumnSpacing = 10, RowSpacing = 5, Margin = 5 };

            smallGrid.Attach(new Label("Min Vertices:") { Halign = Align.Start }, 0, 0, 1, 1);

            _minComponentSizeSpin = new SpinButton(1, 10000, 10);

            _minComponentSizeSpin.Value = 100;

            smallGrid.Attach(_minComponentSizeSpin, 1, 0, 1, 1);

            ((Box)smallFrame.Child).PackStart(smallGrid, false, false, 0);

            optionsBox.PackStart(smallFrame, false, false, 0);



            // Remove Statistical Outliers

            var outlierFrame = CreateOperationFrame("Remove Statistical Outliers",

                "Removes noise vertices based on distance to neighbors",

                out _removeStatisticalOutliers);

            var outlierGrid = new Grid { ColumnSpacing = 10, RowSpacing = 5, Margin = 5 };

            outlierGrid.Attach(new Label("K Neighbors:") { Halign = Align.Start }, 0, 0, 1, 1);

            _kNeighborsSpin = new SpinButton(3, 50, 1);

            _kNeighborsSpin.Value = 10;

            outlierGrid.Attach(_kNeighborsSpin, 1, 0, 1, 1);

            outlierGrid.Attach(new Label("Std Ratio:") { Halign = Align.Start }, 0, 1, 1, 1);

            _stdRatioSpin = new SpinButton(0.5, 5.0, 0.1);

            _stdRatioSpin.Value = 2.0;

            _stdRatioSpin.Digits = 1;

            outlierGrid.Attach(_stdRatioSpin, 1, 1, 1, 1);

            ((Box)outlierFrame.Child).PackStart(outlierGrid, false, false, 0);

            optionsBox.PackStart(outlierFrame, false, false, 0);



            // Fix Non-Manifold

            var manifoldFrame = CreateOperationFrame("Fix Non-Manifold Geometry",

                "Fixes edges shared by more than 2 faces by duplicating vertices",

                out _fixNonManifold);

            optionsBox.PackStart(manifoldFrame, false, false, 0);



            // Fill Holes

            var holesFrame = CreateOperationFrame("Fill Holes",

                "Fills open boundaries using ear clipping triangulation",

                out _fillHoles);

            var holesGrid = new Grid { ColumnSpacing = 10, RowSpacing = 5, Margin = 5 };

            holesGrid.Attach(new Label("Max Hole Edges:") { Halign = Align.Start }, 0, 0, 1, 1);

            _maxHoleEdgesSpin = new SpinButton(3, 500, 10);

            _maxHoleEdgesSpin.Value = 100;

            holesGrid.Attach(_maxHoleEdgesSpin, 1, 0, 1, 1);

            ((Box)holesFrame.Child).PackStart(holesGrid, false, false, 0);

            optionsBox.PackStart(holesFrame, false, false, 0);



            // Fix Normals

            var normalsFrame = CreateOperationFrame("Fix Normal Orientation",

                "Ensures consistent normal direction across mesh faces",

                out _fixNormals);

            optionsBox.PackStart(normalsFrame, false, false, 0);



            // Remove Isolated Vertices

            var isolatedFrame = CreateOperationFrame("Remove Isolated Vertices",

                "Removes vertices not connected to any triangle",

                out _removeIsolatedVertices);

            optionsBox.PackStart(isolatedFrame, false, false, 0);



            scrolled.Add(optionsBox);

            contentArea.PackStart(scrolled, true, true, 0);



            // Connect change handlers

            _removeDegenerateTris.Toggled += OnOptionChanged;

            _removeSlivers.Toggled += OnOptionChanged;

            _removeSmallComponents.Toggled += OnOptionChanged;

            _removeStatisticalOutliers.Toggled += OnOptionChanged;

            _fixNonManifold.Toggled += OnOptionChanged;

            _fillHoles.Toggled += OnOptionChanged;

            _fixNormals.Toggled += OnOptionChanged;

            _removeIsolatedVertices.Toggled += OnOptionChanged;



            // Dialog buttons

            AddButton("Cancel", ResponseType.Cancel);

            AddButton("Apply Cleanup", ResponseType.Ok);



            ShowAll();

        }



        private Frame CreateOperationFrame(string title, string description, out CheckButton checkButton)

        {

            var frame = new Frame();

            var vbox = new Box(Orientation.Vertical, 5);

            vbox.Margin = 5;



            checkButton = new CheckButton(title);

            checkButton.Active = true;

            vbox.PackStart(checkButton, false, false, 0);



            var descLabel = new Label(description);

            descLabel.Halign = Align.Start;

            descLabel.MarginStart = 20;

            var attrs = new Pango.AttrList();

            attrs.Insert(new Pango.AttrScale(0.9));

            attrs.Insert(new Pango.AttrForeground(40000, 40000, 40000));

            descLabel.Attributes = attrs;

            vbox.PackStart(descLabel, false, false, 0);



            frame.Add(vbox);

            return frame;

        }



        private void OnPresetChanged(object? sender, EventArgs e)

        {

            LoadPreset(_presetCombo.ActiveText ?? "Default");

        }



        private void OnOptionChanged(object? sender, EventArgs e)

        {

            // When user changes options manually, switch to Custom preset

            if (_presetCombo.ActiveText != "Custom")

            {

                _presetCombo.Active = 4; // Custom

            }

        }



        private void LoadPreset(string preset)

        {

            switch (preset)

            {

                case "Default":

                    _removeDegenerateTris.Active = true;

                    _removeSlivers.Active = true;

                    _removeSmallComponents.Active = true;

                    _removeStatisticalOutliers.Active = false;

                    _fixNonManifold.Active = true;

                    _fillHoles.Active = false;

                    _fixNormals.Active = true;

                    _removeIsolatedVertices.Active = true;

                    _minComponentSizeSpin.Value = 100;

                    break;



                case "Conservative":

                    _removeDegenerateTris.Active = true;

                    _removeSlivers.Active = false;

                    _removeSmallComponents.Active = false;

                    _removeStatisticalOutliers.Active = false;

                    _fixNonManifold.Active = false;

                    _fillHoles.Active = false;

                    _fixNormals.Active = false;

                    _removeIsolatedVertices.Active = true;

                    break;



                case "Aggressive":

                    _removeDegenerateTris.Active = true;

                    _removeSlivers.Active = true;

                    _removeSmallComponents.Active = true;

                    _removeStatisticalOutliers.Active = true;

                    _fixNonManifold.Active = true;

                    _fillHoles.Active = true;

                    _fixNormals.Active = true;

                    _removeIsolatedVertices.Active = true;

                    _minComponentSizeSpin.Value = 500;

                    _stdRatioSpin.Value = 1.5;

                    break;



                case "Scan Cleanup":

                    _removeDegenerateTris.Active = true;

                    _removeSlivers.Active = true;

                    _removeSmallComponents.Active = true;

                    _removeStatisticalOutliers.Active = true;

                    _fixNonManifold.Active = true;

                    _fillHoles.Active = false;

                    _fixNormals.Active = true;

                    _removeIsolatedVertices.Active = true;

                    _minComponentSizeSpin.Value = 200;

                    _stdRatioSpin.Value = 2.0;

                    _kNeighborsSpin.Value = 15;

                    break;

            }

        }



        protected override void OnResponse(ResponseType response_id)

        {

            if (response_id == ResponseType.Ok)

            {

                Options = new MeshCleanupOptions

                {

                    RemoveDegenerateTris = _removeDegenerateTris.Active,

                    MinTriangleArea = (float)_minTriangleAreaSpin.Value,

                    RemoveSlivers = _removeSlivers.Active,

                    MinAspectRatio = (float)_minAspectRatioSpin.Value,

                    RemoveSmallComponents = _removeSmallComponents.Active,

                    MinComponentSize = (int)_minComponentSizeSpin.Value,

                    RemoveStatisticalOutliers = _removeStatisticalOutliers.Active,

                    KNeighbors = (int)_kNeighborsSpin.Value,

                    StdRatio = (float)_stdRatioSpin.Value,

                    FixNonManifold = _fixNonManifold.Active,

                    FillHoles = _fillHoles.Active,

                    MaxHoleEdges = (int)_maxHoleEdgesSpin.Value,

                    FixNormals = _fixNormals.Active,

                    RemoveIsolatedVertices = _removeIsolatedVertices.Active

                };

            }

            base.OnResponse(response_id);

        }

    }

}