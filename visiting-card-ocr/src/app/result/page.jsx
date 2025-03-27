'use client';
import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import '../globals.css';

export default function ResultPage() {
  const searchParams = useSearchParams();
  const [info, setInfo] = useState(null);
  const [showToast, setShowToast] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const data = searchParams.get('data');
    const success = searchParams.get('success');
    
    if (data) {
      try {
        const parsed = JSON.parse(decodeURIComponent(data));
        setInfo(parsed);
      } catch {
        console.error('Invalid result data');
      }
    }

    if (success === 'true') {
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3000);
    }

  }, [searchParams]);

  if (!info) return null;

  return (
    <main className="upload-container">
      {showToast && <div className="toast">Successfully saved to MongoDB!</div>}

      <h1 className="upload-title">EXTRACTED INFORMATION</h1>
      <div className="upload-output">
        {Object.entries(info).map(([key, value]) => (
          <p key={key}>
            <strong className="field-label">{key}:</strong>{' '}
            {Array.isArray(value) ? value.join(', ') : value}
          </p>
        ))}
      </div>
    </main>
  );
}