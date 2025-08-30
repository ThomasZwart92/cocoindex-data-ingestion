'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Dashboard' },
    { href: '/queue', label: 'Queue' },
    { href: '/settings', label: 'Settings' },
    { href: '/styling', label: 'Styling' },
  ];

  return (
    <nav style={{ borderBottom: '1px solid #E5E7EB' }}>
      <div style={{ maxWidth: '85%', margin: '0 auto' }}>
        <div className="flex gap-2">
          {navItems.map((item, index) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="hover:bg-gray-50"
                style={{
                  paddingTop: '12px',
                  paddingBottom: '12px',
                  paddingLeft: index === 0 ? '0' : '20px',
                  paddingRight: '20px',
                  borderBottom: isActive ? '2px solid #3498DB' : '2px solid transparent',
                  color: isActive ? '#3498DB' : '#2C3E50',
                  backgroundColor: isActive ? '#F8FBFF' : 'transparent',
                  fontSize: '14px',
                  fontWeight: '500',
                  transition: 'all 0.2s',
                }}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}