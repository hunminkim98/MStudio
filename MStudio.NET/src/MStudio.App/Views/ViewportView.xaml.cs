using System.Collections.Specialized;
using System.Numerics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using HelixToolkit.Maths;
using HelixToolkit.SharpDX;
using MStudio.App.ViewModels;
using HelixToolkit.Wpf.SharpDX;

namespace MStudio.App.Views
{
    /// <summary>
    /// ViewportView - 3D viewport for motion capture visualization.
    /// 
    /// Clean Architecture Notes:
    /// - View handles low-level UI events (mouse clicks, color picker dialog)
    /// - Converts screen coordinates to 3D picking request
    /// - Delegates marker selection logic to ViewModel
    /// </summary>
    public partial class ViewportView : UserControl
    {
        public ViewportView()
        {
            InitializeComponent();
            DataContextChanged += OnDataContextChanged;
        }

        private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.OldValue is MStudioViewportViewModel oldVm)
            {
                oldVm.SceneElements.CollectionChanged -= OnSceneElementsChanged;
                viewport.Items.Clear();
            }

            if (e.NewValue is MStudioViewportViewModel newVm)
            {
                // Add existing elements
                foreach (var element in newVm.SceneElements)
                {
                    viewport.Items.Add(element);
                }

                // Subscribe to future changes
                newVm.SceneElements.CollectionChanged += OnSceneElementsChanged;
            }
        }

        private void OnSceneElementsChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            switch (e.Action)
            {
                case NotifyCollectionChangedAction.Add:
                    if (e.NewItems != null)
                    {
                        foreach (Element3D item in e.NewItems)
                        {
                            viewport.Items.Add(item);
                        }
                    }
                    break;
                case NotifyCollectionChangedAction.Remove:
                    if (e.OldItems != null)
                    {
                        foreach (Element3D item in e.OldItems)
                        {
                            viewport.Items.Remove(item);
                        }
                    }
                    break;
                case NotifyCollectionChangedAction.Reset:
                    viewport.Items.Clear();
                    break;
            }
        }

        /// <summary>
        /// Handles mouse click for marker picking.
        /// Casts a ray from the camera through the click point and finds the nearest marker.
        /// </summary>
        private void Viewport_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (DataContext is not MStudioViewportViewModel vm)
                return;

            // Get click position relative to viewport
            var clickPoint = e.GetPosition(viewport);

            // Get the render context to access viewport matrices
            var renderContext = viewport.RenderContext;
            if (renderContext == null)
                return;

            // Use HelixToolkit's UnProject extension method
            // This creates a ray from the camera through the clicked point
            var screenPos = new Vector2((float)clickPoint.X, (float)clickPoint.Y);
            
            // UnProject returns a Ray from the camera through the screen point
            viewport.UnProject(screenPos, out var ray);

            // Convert to System.Numerics.Vector3
            var rayOrigin = new System.Numerics.Vector3(ray.Position.X, ray.Position.Y, ray.Position.Z);
            var rayDirection = new System.Numerics.Vector3(ray.Direction.X, ray.Direction.Y, ray.Direction.Z);

            // Ask ViewModel to select the nearest marker along this ray
            bool markerSelected = vm.SelectMarkerByRay(rayOrigin, rayDirection);

            // Only mark as handled if we actually selected a marker
            // This allows camera rotation to work when clicking on empty space
            if (markerSelected)
            {
                e.Handled = true;
            }
        }

        /// <summary>
        /// Opens color picker dialog for marker color selection.
        /// 
        /// Clean Architecture: View handles UI dialog, ViewModel receives the result.
        /// </summary>
        private void MarkerColorButton_Click(object sender, RoutedEventArgs e)
        {
            if (DataContext is not MStudioViewportViewModel vm)
                return;

            // Use Windows Forms ColorDialog (WPF doesn't have a built-in one)
            using var colorDialog = new System.Windows.Forms.ColorDialog
            {
                Color = System.Drawing.Color.FromArgb(
                    (int)(vm.MarkerColor.Alpha * 255),
                    (int)(vm.MarkerColor.Red * 255),
                    (int)(vm.MarkerColor.Green * 255),
                    (int)(vm.MarkerColor.Blue * 255)),
                FullOpen = true,
                AllowFullOpen = true
            };

            if (colorDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var c = colorDialog.Color;
                vm.MarkerColor = new Color4(c.R / 255f, c.G / 255f, c.B / 255f, vm.MarkerColor.Alpha);
            }
        }

        /// <summary>
        /// Opens color picker dialog for bone color selection.
        /// 
        /// Clean Architecture: View handles UI dialog, ViewModel receives the result.
        /// </summary>
        private void BoneColorButton_Click(object sender, RoutedEventArgs e)
        {
            if (DataContext is not MStudioViewportViewModel vm)
                return;

            using var colorDialog = new System.Windows.Forms.ColorDialog
            {
                Color = System.Drawing.Color.FromArgb(
                    vm.BoneColor.A,
                    vm.BoneColor.R,
                    vm.BoneColor.G,
                    vm.BoneColor.B),
                FullOpen = true,
                AllowFullOpen = true
            };

            if (colorDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var c = colorDialog.Color;
                vm.BoneColor = System.Windows.Media.Color.FromArgb(vm.BoneColor.A, c.R, c.G, c.B);
            }
        }
    }
}
