using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using MStudio.App.ViewModels;

namespace MStudio.App.Views
{
    public partial class GraphView : UserControl
    {
        private bool _isDragging;

        public GraphView()
        {
            InitializeComponent();
            SizeChanged += OnSizeChanged;
            Loaded += OnLoaded;
            
            // Mouse events for seeking (on the graph area)
            MouseLeftButtonDown += OnMouseLeftButtonDown;
            MouseMove += OnMouseMove;
            MouseLeftButtonUp += OnMouseLeftButtonUp;
            MouseLeave += OnMouseLeave;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            UpdateViewModelSize();
        }

        private void OnSizeChanged(object sender, SizeChangedEventArgs e)
        {
            UpdateViewModelSize();
        }

        private void UpdateViewModelSize()
        {
            if (DataContext is GraphViewModel vm && GraphArea.ActualWidth > 0 && GraphArea.ActualHeight > 0)
            {
                vm.ViewWidth = GraphArea.ActualWidth;
                vm.ViewHeight = GraphArea.ActualHeight;
                vm.UpdatePoints();
            }
        }

        private void OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (DataContext is GraphViewModel vm)
            {
                _isDragging = true;
                CaptureMouse();
                
                // Get position relative to GraphArea (excluding Y-axis label area)
                var pos = e.GetPosition(GraphArea);
                vm.SeekToPosition(pos.X);
                
                e.Handled = true;
            }
        }

        private void OnMouseMove(object sender, MouseEventArgs e)
        {
            if (_isDragging && DataContext is GraphViewModel vm)
            {
                var pos = e.GetPosition(GraphArea);
                vm.SeekToPosition(pos.X);
            }
        }

        private void OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (_isDragging)
            {
                _isDragging = false;
                ReleaseMouseCapture();
                e.Handled = true;
            }
        }

        private void OnMouseLeave(object sender, MouseEventArgs e)
        {
            // Don't stop dragging on leave - we have mouse capture
            // This allows dragging outside the control bounds
        }
    }
}

