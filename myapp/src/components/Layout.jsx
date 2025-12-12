import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import Chatbox from './Chatbox';

const Layout = () => {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <Chatbox />
      </div>
    </div>
  );
};

export default Layout;
