async () => {
  for (let i = 0; i < 80 && !globalThis.__answerPreview; i++) await new Promise(r => setTimeout(r, 100));
  const dialog = document.getElementById('workspace-tour');
  // Exercise the actual native-close path used by preview startup, without bypassing the controller.
  document.getElementById('tour-replay').click();
  if (!dialog.open) throw new Error('TOUR_NOT_READY_FOR_SCROLL_REPRO');
  dialog.close();
  await new Promise(r => setTimeout(r, 50));
  const overflow = getComputedStyle(document.body).overflowY;
  const result = { kind: 'FRESH_LOCAL_PREVIEW_MAIN_SCROLL', dialog_open: dialog.open,
    body_class: document.body.className, body_overflow_y: overflow,
    pass: dialog.open || !['hidden', 'clip'].includes(overflow), aws_calls: 0, provider_calls: 0 };
  globalThis.__mainScrollCheck = result;
  if (!result.pass) throw new Error('MAIN_SCROLL_LOCK_WITHOUT_OPEN_DIALOG:' + JSON.stringify(result));
  const checks = [{ name: 'native close releases page scroll', pass: result.pass }];
  const check = (name, pass) => { checks.push({ name, pass: !!pass }); if (!pass) throw new Error(name); };
  const replay = document.getElementById('tour-replay');
  replay.click(); dialog.close(); replay.click();
  await new Promise(r => setTimeout(r, 50));
  check('late close event keeps a newly opened tour locked', dialog.open && document.body.classList.contains('tour-open'));
  document.getElementById('tour-skip').click();
  await new Promise(r => setTimeout(r, 50));
  check('Skip releases lock and highlight', !dialog.open && !document.body.classList.contains('tour-open') && !document.querySelector('.tour-target'));
  replay.click(); dialog.close(); __preview.ui.clear();
  check('deactivation after native close removes orphan lock immediately', !document.body.classList.contains('tour-open'));
  await __preview.ui.activate();
  check('reactivated workspace has no hidden closed-dialog lock', dialog.open || !['hidden', 'clip'].includes(getComputedStyle(document.body).overflowY));
  return { ...result, checks, passed: checks.length, url: location.href };
}
