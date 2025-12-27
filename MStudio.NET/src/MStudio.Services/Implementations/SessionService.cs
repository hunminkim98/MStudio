using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using MStudio.Core.Models;
using MStudio.Core.Parsers;
using MStudio.Services.Interfaces;

namespace MStudio.Services.Implementations
{
    public class SessionService : ISessionService
    {
        private readonly IEnumerable<IFileParser> _parsers;
        private MotionData? _currentMotion;
        private string? _currentFilePath;
        private bool _isLoading;

        public SessionService(IEnumerable<IFileParser> parsers)
        {
            _parsers = parsers;
        }

        public MotionData? CurrentMotion
        {
            get => _currentMotion;
            private set => SetProperty(ref _currentMotion, value);
        }

        public string? CurrentFilePath
        {
            get => _currentFilePath;
            private set => SetProperty(ref _currentFilePath, value);
        }

        public bool IsLoading
        {
            get => _isLoading;
            private set => SetProperty(ref _isLoading, value);
        }

        public async Task LoadMotionAsync(string filePath)
        {
            var parser = _parsers.FirstOrDefault(p => p.CanParse(filePath));
            if (parser == null) throw new NotSupportedException($"No parser found for file: {filePath}");

            IsLoading = true;
            try
            {
                var data = await parser.ParseAsync(filePath);
                CurrentFilePath = filePath; // Set path before motion to ensure UI updates correctly
                CurrentMotion = data;
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Synchronous file loading wrapper
        /// </summary>
        public void LoadFile(string filePath)
        {
            // Run async method synchronously (acceptable for UI-driven operations)
            LoadMotionAsync(filePath).GetAwaiter().GetResult();
        }

        public void CloseSession()
        {
            CurrentMotion?.Markers.Dispose();
            CurrentMotion = null;
            CurrentFilePath = null;
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(storage, value)) return false;
            storage = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }
}
