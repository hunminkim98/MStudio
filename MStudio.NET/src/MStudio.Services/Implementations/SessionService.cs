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
                CurrentMotion = data;
            }
            finally
            {
                IsLoading = false;
            }
        }

        public void CloseSession()
        {
            CurrentMotion?.Markers.Dispose();
            CurrentMotion = null;
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
