using System.Windows;

namespace MStudio.App.Behaviors
{
    /// <summary>
    /// Attached behavior that observes a FrameworkElement's ActualWidth and ActualHeight
    /// and exposes them as bindable properties.
    /// 
    /// This allows clean MVVM binding without code-behind:
    /// - View's actual size is passed to ViewModel via OneWayToSource binding
    /// - ViewModel can react to size changes via property change notifications
    /// 
    /// Usage in XAML:
    /// <Grid behaviors:SizeObserverBehavior.IsEnabled="True"
    ///       behaviors:SizeObserverBehavior.ObservedWidth="{Binding ViewWidth, Mode=OneWayToSource}"
    ///       behaviors:SizeObserverBehavior.ObservedHeight="{Binding ViewHeight, Mode=OneWayToSource}"/>
    /// 
    /// Architecture: View Infrastructure Layer (Attached Behavior Pattern)
    /// </summary>
    public static class SizeObserverBehavior
    {
        #region IsEnabled Attached Property

        public static readonly DependencyProperty IsEnabledProperty =
            DependencyProperty.RegisterAttached(
                "IsEnabled",
                typeof(bool),
                typeof(SizeObserverBehavior),
                new PropertyMetadata(false, OnIsEnabledChanged));

        public static bool GetIsEnabled(DependencyObject obj) =>
            (bool)obj.GetValue(IsEnabledProperty);

        public static void SetIsEnabled(DependencyObject obj, bool value) =>
            obj.SetValue(IsEnabledProperty, value);

        #endregion

        #region ObservedWidth Attached Property

        public static readonly DependencyProperty ObservedWidthProperty =
            DependencyProperty.RegisterAttached(
                "ObservedWidth",
                typeof(double),
                typeof(SizeObserverBehavior),
                new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

        public static double GetObservedWidth(DependencyObject obj) =>
            (double)obj.GetValue(ObservedWidthProperty);

        public static void SetObservedWidth(DependencyObject obj, double value) =>
            obj.SetValue(ObservedWidthProperty, value);

        #endregion

        #region ObservedHeight Attached Property

        public static readonly DependencyProperty ObservedHeightProperty =
            DependencyProperty.RegisterAttached(
                "ObservedHeight",
                typeof(double),
                typeof(SizeObserverBehavior),
                new FrameworkPropertyMetadata(0.0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

        public static double GetObservedHeight(DependencyObject obj) =>
            (double)obj.GetValue(ObservedHeightProperty);

        public static void SetObservedHeight(DependencyObject obj, double value) =>
            obj.SetValue(ObservedHeightProperty, value);

        #endregion

        #region Event Handling

        private static void OnIsEnabledChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is not FrameworkElement element)
                return;

            if ((bool)e.NewValue)
            {
                element.SizeChanged += OnSizeChanged;
                element.Loaded += OnLoaded;
                
                // Update immediately if already loaded
                if (element.IsLoaded)
                {
                    UpdateObservedSize(element);
                }
            }
            else
            {
                element.SizeChanged -= OnSizeChanged;
                element.Loaded -= OnLoaded;
            }
        }

        private static void OnLoaded(object sender, RoutedEventArgs e)
        {
            if (sender is FrameworkElement element)
            {
                UpdateObservedSize(element);
            }
        }

        private static void OnSizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (sender is FrameworkElement element)
            {
                UpdateObservedSize(element);
            }
        }

        private static void UpdateObservedSize(FrameworkElement element)
        {
            if (element.ActualWidth > 0)
            {
                SetObservedWidth(element, element.ActualWidth);
            }
            if (element.ActualHeight > 0)
            {
                SetObservedHeight(element, element.ActualHeight);
            }
        }

        #endregion
    }
}
