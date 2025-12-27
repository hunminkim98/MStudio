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
    /// - View handles low-level UI events (mouse clicks)
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
    }
}

