// src/components/Header.js
import React from 'react';
import logo from '../images/logo.png';
import text from '../images/text.png';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <img src={logo} alt="Echo Hall" className="header-logo" />
        <img src={text} alt="Echo Hall Text" className="header-text" />
      </div>
    </header>
  );
}

export default Header;