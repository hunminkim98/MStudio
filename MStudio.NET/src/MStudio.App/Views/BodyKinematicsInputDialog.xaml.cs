using System.Globalization;
using System.Windows;
using System.Windows.Input;

namespace MStudio.App.Views
{
    public partial class BodyKinematicsInputDialog : Window
    {
        public float BodyMassKg { get; private set; } = 70f;
        public float HeightM { get; private set; } = 1.70f;
        public bool Confirmed { get; private set; } = false;

        public BodyKinematicsInputDialog()
        {
            InitializeComponent();
        }

        private void Header_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        private void RunButton_Click(object sender, RoutedEventArgs e)
        {
            // Parse mass
            if (!float.TryParse(MassInput.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float mass) || mass <= 0)
            {
                MessageBox.Show("Please enter a valid body mass in kg.", "Invalid Input", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Parse height
            if (!float.TryParse(HeightInput.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float height) || height <= 0)
            {
                MessageBox.Show("Please enter a valid height in meters.", "Invalid Input", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            BodyMassKg = mass;
            HeightM = height;
            Confirmed = true;
            DialogResult = true;
            Close();
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
