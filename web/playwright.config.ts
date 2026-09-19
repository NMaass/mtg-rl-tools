import { defineConfig } from '@playwright/test';
export default defineConfig({testDir:'./tests/browser',fullyParallel:false,workers:1,retries:0,timeout:120000,use:{baseURL:'http://127.0.0.1:4173',viewport:{width:1440,height:960},reducedMotion:'reduce',trace:'retain-on-failure'},webServer:{command:'npm run preview -- --port 4173',url:'http://127.0.0.1:4173',reuseExistingServer:!process.env.CI}});
