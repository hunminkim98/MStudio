using System.Windows;
using System.Windows.Input;

namespace MStudio.App.Behaviors
{
    /// <summary>
    /// Global keyboard behavior that intercepts specific keys at the Window level
    /// before any child controls can handle them.
    /// 
    /// This ensures Left/Right/Space keys are always used for timeline navigation
    /// regardless of which control has focus (ComboBox, ListBox, etc.)
    /// 
    /// Architecture: View Infrastructure Layer (Attached Behavior Pattern)
    /// - Captures UI events (PreviewKeyDown - tunneling)
    /// - Routes to ViewModel commands via binding
    /// - No business logic here - only event interception
    /// </summary>
    public static class GlobalKeyboardBehavior
    {
        #region IsEnabled Attached Property

        public static readonly DependencyProperty IsEnabledProperty =
            DependencyProperty.RegisterAttached(
                "IsEnabled",
                typeof(bool),
                typeof(GlobalKeyboardBehavior),
                new PropertyMetadata(false, OnIsEnabledChanged));

        public static bool GetIsEnabled(DependencyObject obj) =>
            (bool)obj.GetValue(IsEnabledProperty);

        public static void SetIsEnabled(DependencyObject obj, bool value) =>
            obj.SetValue(IsEnabledProperty, value);

        #endregion

        #region LeftKeyCommand Attached Property

        public static readonly DependencyProperty LeftKeyCommandProperty =
            DependencyProperty.RegisterAttached(
                "LeftKeyCommand",
                typeof(ICommand),
                typeof(GlobalKeyboardBehavior),
                new PropertyMetadata(null));

        public static ICommand GetLeftKeyCommand(DependencyObject obj) =>
            (ICommand)obj.GetValue(LeftKeyCommandProperty);

        public static void SetLeftKeyCommand(DependencyObject obj, ICommand value) =>
            obj.SetValue(LeftKeyCommandProperty, value);

        #endregion

        #region RightKeyCommand Attached Property

        public static readonly DependencyProperty RightKeyCommandProperty =
            DependencyProperty.RegisterAttached(
                "RightKeyCommand",
                typeof(ICommand),
                typeof(GlobalKeyboardBehavior),
                new PropertyMetadata(null));

        public static ICommand GetRightKeyCommand(DependencyObject obj) =>
            (ICommand)obj.GetValue(RightKeyCommandProperty);

        public static void SetRightKeyCommand(DependencyObject obj, ICommand value) =>
            obj.SetValue(RightKeyCommandProperty, value);

        #endregion

        #region SpaceKeyCommand Attached Property

        public static readonly DependencyProperty SpaceKeyCommandProperty =
            DependencyProperty.RegisterAttached(
                "SpaceKeyCommand",
                typeof(ICommand),
                typeof(GlobalKeyboardBehavior),
                new PropertyMetadata(null));

        public static ICommand GetSpaceKeyCommand(DependencyObject obj) =>
            (ICommand)obj.GetValue(SpaceKeyCommandProperty);

        public static void SetSpaceKeyCommand(DependencyObject obj, ICommand value) =>
            obj.SetValue(SpaceKeyCommandProperty, value);

        #endregion

        #region Event Handling

        private static void OnIsEnabledChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is not Window window)
                return;

            if ((bool)e.NewValue)
            {
                // Use PreviewKeyDown (tunneling) to intercept BEFORE child controls
                window.PreviewKeyDown += OnPreviewKeyDown;
            }
            else
            {
                window.PreviewKeyDown -= OnPreviewKeyDown;
            }
        }

        private static void OnPreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (sender is not Window window)
                return;

            // Only intercept our global navigation keys
            switch (e.Key)
            {
                case Key.Left:
                    ExecuteCommand(window, LeftKeyCommandProperty);
                    e.Handled = true; // Prevent ComboBox/ListBox from handling
                    break;

                case Key.Right:
                    ExecuteCommand(window, RightKeyCommandProperty);
                    e.Handled = true;
                    break;

                case Key.Space:
                    // Don't intercept Space if focus is on a TextBox (user might be typing)
                    if (Keyboard.FocusedElement is System.Windows.Controls.TextBox)
                        return;
                    
                    ExecuteCommand(window, SpaceKeyCommandProperty);
                    e.Handled = true;
                    break;
            }
        }

        private static void ExecuteCommand(Window window, DependencyProperty commandProperty)
        {
            var command = (ICommand)window.GetValue(commandProperty);
            if (command?.CanExecute(null) == true)
            {
                command.Execute(null);
            }
        }

        #endregion
    }
}
