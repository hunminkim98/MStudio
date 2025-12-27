using System;
using System.Windows;
using Microsoft.Extensions.DependencyInjection;
using MStudio.App.Services;
using MStudio.App.ViewModels;
using MStudio.Core.Interfaces;
using MStudio.Core.Parsers;
using MStudio.Services;
using MStudio.Services.Implementations;
using MStudio.Services.Interfaces;

namespace MStudio.App
{
    public partial class App : Application
    {
        public static IServiceProvider ServiceProvider { get; private set; } = null!;

        protected override void OnStartup(StartupEventArgs e)
        {
            try
            {
                var serviceCollection = new ServiceCollection();
                ConfigureServices(serviceCollection);

                ServiceProvider = serviceCollection.BuildServiceProvider();

                var mainWindow = ServiceProvider.GetRequiredService<MainWindow>();
                mainWindow.DataContext = ServiceProvider.GetRequiredService<MainViewModel>();
                mainWindow.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Startup Error: {ex.Message}\n\nStack Trace: {ex.StackTrace}", "MStudio Startup Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Shutdown();
            }
        }

        private void ConfigureServices(IServiceCollection services)
        {
            // Core Parsers
            services.AddSingleton<IFileParser, TrcFileParser>();
            services.AddSingleton<IFileParser, C3dFileParser>();
            services.AddSingleton<IFileParser, JsonPoseParser>();

            // Services
            services.AddSingleton<ISessionService, SessionService>();
            services.AddSingleton<ITimelineService, TimelineService>();
            services.AddSingleton<IVisualizationSettingsService, VisualizationSettingsService>();
            
            // UI Services (Clean Architecture: platform-specific implementations in App layer)
            services.AddSingleton<IDialogService, DialogService>();

            // ViewModels / Windows
            services.AddSingleton<MStudioViewportViewModel>();
            services.AddSingleton<GraphViewModel>();
            services.AddSingleton<DataViewModel>();
            services.AddSingleton<MainViewModel>();
            services.AddTransient<MainWindow>();
        }
    }
}
