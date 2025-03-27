'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { FiUpload, FiFile } from 'react-icons/fi';
import './globals.css';

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [title, setTitle] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  const router = useRouter();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) setSelectedFile(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) setSelectedFile(file);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile || !title.trim()) return alert('Please provide a title and select a file.');

    setUploading(true);
    setSaveStatus(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // 1. Send image to OCR backend
      const res = await fetch('http://localhost:8000/api/ocr', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      const finalData = { ...data, title: title.trim() };

      // 2. Save to MongoDB
      const save = await fetch('/api/cards', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalData),
      });

      const result = await save.json();
      setSaveStatus(result.success ? 'success' : 'error');

      // 3. Redirect to result page
      if (result.success) {
        const encoded = encodeURIComponent(JSON.stringify(finalData));
        router.push(`/result?data=${encoded}`);
      }
    } catch (error) {
      console.error('Upload Error:', error);
      setSaveStatus('error');
      alert('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="upload-container">
      <h1 className="upload-title">ADD CARD</h1>
      <p className="upload-subtitle">Upload the visiting card and extract information from it</p>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="title-input"
        />

        <div
          className={`upload-dropzone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          <div className="floating-icon">
            <FiFile size={32} />
          </div>
          <p className="drop-label">Drag and drop an image</p>
          <p className="drop-separator">or</p>
          <label htmlFor="fileInput" className="upload-button">
            Browse files
          </label>
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            className="upload-hidden"
          />
        </div>

        {selectedFile && (
          <p className="upload-selected">
            <strong>{selectedFile.name}</strong>
          </p>
        )}

        {uploading && <div className="upload-spinner"></div>}

        <button type="submit" className="upload-submit" disabled={uploading}>
          <FiUpload style={{ marginRight: '8px' }} />
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </form>

      {saveStatus === 'error' && (
        <div className="alert error">Failed to save to database.</div>
      )}
      {saveStatus === 'success' && (
        <div className="toast">Successfully saved to MongoDB!</div>
      )}
    </main>
  );
}