import { test, expect } from '@playwright/test';
import { example } from '../../src/example';
const id = 'c'.repeat(64);

test('public recordings keep both sides visible and transport fits a laptop viewport', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  const replay = structuredClone(example);
  replay.source = 'mtgo';
  replay.title = 'Public game';
  replay.frames = replay.frames.slice(0, 2).map(frame => {
    const view = frame.views['1'];
    view.viewer = 'public'; view.priority = null;
    view.players.forEach(player => { player.hand = []; });
    return { ...frame, views: { public: view }, decisions: {} };
  });
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/session') return route.fulfill({ json: { hasKey: false } });
    if (path === '/api/replays') return route.fulfill({ json: [{ id, title: replay.title, source: 'mtgo', frames: 2, created_at: '2026-09-19' }] });
    if (path.endsWith('/report')) return route.fulfill({ json: { ratings: [], attempts: [] } });
    return route.fulfill({ json: replay });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open Public game', exact: true }).click();
  await expect(page.locator('.hero-half .game-card')).toHaveCount(2);
  await expect(page.locator('.opponent-half .game-card')).toHaveCount(2);
  await expect(page.locator('.hero-half .player-bar')).toContainText('Player 1');
  await expect(page.locator('.opponent-half .player-bar')).toContainText('Player 2');
  await expect(page.locator('.hand-area')).toContainText('PUBLIC RECORDING');
  const transport = await page.locator('.transport').boundingBox();
  expect(transport).not.toBeNull();
  expect(transport!.y + transport!.height).toBeLessThanOrEqual(800);
  await page.getByRole('button', { name: 'Next position', exact: true }).click();
  await expect(page.locator('.position-count')).toHaveText('2 / 2');
  expect(await page.locator('.transport').boundingBox()).toEqual(transport);
});
