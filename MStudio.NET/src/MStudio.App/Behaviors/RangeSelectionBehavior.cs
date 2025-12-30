using System.Windows;
using System.Windows.Input;

namespace MStudio.App.Behaviors
{
    /// <summary>
    /// Attached behavior for handling range selection on the graph via mouse drag.
    /// When enabled, RIGHT mouse button drag creates a frame range selection.
    /// Left mouse button is reserved for seeking/frame navigation.
    /// </summary>
    public static class RangeSelectionBehavior
    {
        #region IsEnabled Property

        public static readonly DependencyProperty IsEnabledProperty =
            DependencyProperty.RegisterAttached(
                "IsEnabled",
                typeof(bool),
                typeof(RangeSelectionBehavior),
                new PropertyMetadata(false, OnIsEnabledChanged));

        public static bool GetIsEnabled(DependencyObject obj) => (bool)obj.GetValue(IsEnabledProperty);
        public static void SetIsEnabled(DependencyObject obj, bool value) => obj.SetValue(IsEnabledProperty, value);

        #endregion

        #region StartSelectionCommand Property

        public static readonly DependencyProperty StartSelectionCommandProperty =
            DependencyProperty.RegisterAttached(
                "StartSelectionCommand",
                typeof(ICommand),
                typeof(RangeSelectionBehavior));

        public static ICommand GetStartSelectionCommand(DependencyObject obj) => (ICommand)obj.GetValue(StartSelectionCommandProperty);
        public static void SetStartSelectionCommand(DependencyObject obj, ICommand value) => obj.SetValue(StartSelectionCommandProperty, value);

        #endregion

        #region UpdateSelectionCommand Property

        public static readonly DependencyProperty UpdateSelectionCommandProperty =
            DependencyProperty.RegisterAttached(
                "UpdateSelectionCommand",
                typeof(ICommand),
                typeof(RangeSelectionBehavior));

        public static ICommand GetUpdateSelectionCommand(DependencyObject obj) => (ICommand)obj.GetValue(UpdateSelectionCommandProperty);
        public static void SetUpdateSelectionCommand(DependencyObject obj, ICommand value) => obj.SetValue(UpdateSelectionCommandProperty, value);

        #endregion

        #region EndSelectionCommand Property

        public static readonly DependencyProperty EndSelectionCommandProperty =
            DependencyProperty.RegisterAttached(
                "EndSelectionCommand",
                typeof(ICommand),
                typeof(RangeSelectionBehavior));

        public static ICommand GetEndSelectionCommand(DependencyObject obj) => (ICommand)obj.GetValue(EndSelectionCommandProperty);
        public static void SetEndSelectionCommand(DependencyObject obj, ICommand value) => obj.SetValue(EndSelectionCommandProperty, value);

        #endregion

        private static void OnIsEnabledChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is UIElement element)
            {
                if ((bool)e.NewValue)
                {
                    element.MouseRightButtonDown += OnMouseRightButtonDown;
                    element.MouseMove += OnMouseMove;
                    element.MouseRightButtonUp += OnMouseRightButtonUp;
                    element.MouseLeave += OnMouseLeave;
                }
                else
                {
                    element.MouseRightButtonDown -= OnMouseRightButtonDown;
                    element.MouseMove -= OnMouseMove;
                    element.MouseRightButtonUp -= OnMouseRightButtonUp;
                    element.MouseLeave -= OnMouseLeave;
                }
            }
        }

        private static void OnMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is UIElement element)
            {
                var command = GetStartSelectionCommand(element);
                if (command != null)
                {
                    var position = e.GetPosition(element);
                    if (command.CanExecute(position.X))
                    {
                        command.Execute(position.X);
                        element.CaptureMouse();
                        e.Handled = true;
                    }
                }
            }
        }

        private static void OnMouseMove(object sender, MouseEventArgs e)
        {
            if (sender is UIElement element && e.RightButton == MouseButtonState.Pressed && element.IsMouseCaptured)
            {
                var command = GetUpdateSelectionCommand(element);
                if (command != null)
                {
                    var position = e.GetPosition(element);
                    if (command.CanExecute(position.X))
                    {
                        command.Execute(position.X);
                    }
                }
            }
        }

        private static void OnMouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (sender is UIElement element)
            {
                if (element.IsMouseCaptured)
                {
                    element.ReleaseMouseCapture();
                }

                var command = GetEndSelectionCommand(element);
                if (command != null && command.CanExecute(null))
                {
                    command.Execute(null);
                }
            }
        }

        private static void OnMouseLeave(object sender, MouseEventArgs e)
        {
            if (sender is UIElement element && element.IsMouseCaptured)
            {
                // Don't end selection on mouse leave if still captured
                // Selection will end on mouse up
            }
        }
    }
}
