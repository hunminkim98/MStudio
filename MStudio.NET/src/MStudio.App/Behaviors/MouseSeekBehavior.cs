using System.Windows;
using System.Windows.Input;

namespace MStudio.App.Behaviors
{
    /// <summary>
    /// Attached behavior for mouse-based seeking on a timeline/graph control.
    /// Handles mouse click and drag to seek to a specific position.
    /// 
    /// This allows clean MVVM interaction without code-behind:
    /// - Mouse events are translated to Command invocations
    /// - X position is passed as command parameter
    /// - Dragging state is managed internally
    /// 
    /// Usage in XAML:
    /// <Grid behaviors:MouseSeekBehavior.IsEnabled="True"
    ///       behaviors:MouseSeekBehavior.SeekCommand="{Binding SeekCommand}"
    ///       behaviors:MouseSeekBehavior.ContainerWidth="{Binding ViewWidth}"/>
    /// 
    /// Architecture: View Infrastructure Layer (Attached Behavior Pattern)
    /// </summary>
    public static class MouseSeekBehavior
    {
        // Track dragging state per element
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<UIElement, DragState> _dragStates = new();

        private class DragState
        {
            public bool IsDragging { get; set; }
        }

        #region IsEnabled Attached Property

        public static readonly DependencyProperty IsEnabledProperty =
            DependencyProperty.RegisterAttached(
                "IsEnabled",
                typeof(bool),
                typeof(MouseSeekBehavior),
                new PropertyMetadata(false, OnIsEnabledChanged));

        public static bool GetIsEnabled(DependencyObject obj) =>
            (bool)obj.GetValue(IsEnabledProperty);

        public static void SetIsEnabled(DependencyObject obj, bool value) =>
            obj.SetValue(IsEnabledProperty, value);

        #endregion

        #region SeekCommand Attached Property

        public static readonly DependencyProperty SeekCommandProperty =
            DependencyProperty.RegisterAttached(
                "SeekCommand",
                typeof(ICommand),
                typeof(MouseSeekBehavior),
                new PropertyMetadata(null));

        public static ICommand GetSeekCommand(DependencyObject obj) =>
            (ICommand)obj.GetValue(SeekCommandProperty);

        public static void SetSeekCommand(DependencyObject obj, ICommand value) =>
            obj.SetValue(SeekCommandProperty, value);

        #endregion

        #region RelativeElement Attached Property (optional - for getting position relative to a child element)

        public static readonly DependencyProperty RelativeElementProperty =
            DependencyProperty.RegisterAttached(
                "RelativeElement",
                typeof(UIElement),
                typeof(MouseSeekBehavior),
                new PropertyMetadata(null));

        public static UIElement? GetRelativeElement(DependencyObject obj) =>
            (UIElement?)obj.GetValue(RelativeElementProperty);

        public static void SetRelativeElement(DependencyObject obj, UIElement? value) =>
            obj.SetValue(RelativeElementProperty, value);

        #endregion

        #region Event Handling

        private static void OnIsEnabledChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is not UIElement element)
                return;

            if ((bool)e.NewValue)
            {
                element.MouseLeftButtonDown += OnMouseLeftButtonDown;
                element.MouseMove += OnMouseMove;
                element.MouseLeftButtonUp += OnMouseLeftButtonUp;
                element.MouseLeave += OnMouseLeave;
                
                _dragStates.GetOrCreateValue(element);
            }
            else
            {
                element.MouseLeftButtonDown -= OnMouseLeftButtonDown;
                element.MouseMove -= OnMouseMove;
                element.MouseLeftButtonUp -= OnMouseLeftButtonUp;
                element.MouseLeave -= OnMouseLeave;
                
                _dragStates.Remove(element);
            }
        }

        private static void OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is not UIElement element)
                return;

            if (!_dragStates.TryGetValue(element, out var state))
                return;

            state.IsDragging = true;
            element.CaptureMouse();

            ExecuteSeekCommand(element, e);
            e.Handled = true;
        }

        private static void OnMouseMove(object sender, MouseEventArgs e)
        {
            if (sender is not UIElement element)
                return;

            if (!_dragStates.TryGetValue(element, out var state) || !state.IsDragging)
                return;

            ExecuteSeekCommand(element, e);
        }

        private static void OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (sender is not UIElement element)
                return;

            if (!_dragStates.TryGetValue(element, out var state) || !state.IsDragging)
                return;

            state.IsDragging = false;
            element.ReleaseMouseCapture();
            e.Handled = true;
        }

        private static void OnMouseLeave(object sender, MouseEventArgs e)
        {
            // Don't stop dragging on leave - we have mouse capture
            // This allows dragging outside the control bounds
        }

        private static void ExecuteSeekCommand(UIElement element, MouseEventArgs e)
        {
            var command = GetSeekCommand(element);
            if (command == null)
                return;

            // Get position relative to the specified element or the element itself
            var relativeElement = GetRelativeElement(element) ?? element;
            var position = e.GetPosition(relativeElement);

            if (command.CanExecute(position.X))
            {
                command.Execute(position.X);
            }
        }

        #endregion
    }
}
