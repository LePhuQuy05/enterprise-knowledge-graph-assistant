async () => {
  const checks = [], check = (name, pass) => { checks.push({ name, pass: !!pass }); if (!pass) throw new Error(name); };
  const methods = ['bm25', 'vector', 'hybrid', 'hybrid_reranked'];
  const ids = [...methods, 'hybrid_multi_query', 'hybrid_multi_query_balanced', 'graph_1hop', 'graph_2hop'];
  const original = __preview.overrides.runs, postsBefore = __ownedPreview.posts.length;
  __preview.overrides.runs = async (path, options = {}) => {
    if (path === '/v1/run-capabilities') return { ...(await original(path, options)), methods,
      answering_enabled: false, answer_methods: [], maximum_answer_methods: 1 };
    return original(path, options);
  };
  __preview.ui.clear(); await __preview.ui.activate(); document.querySelector('dialog[open]')?.close();
  for (const id of ids) {
    document.querySelector('[data-page="method-' + id + '"]').click();
    const root = document.getElementById('page-method-' + id);
    for (let i = 0; i < 80 && root.querySelectorAll('.method-step').length !== 6; i++) await new Promise(r => setTimeout(r, 100));
    check(id + ' keeps recorded six-step walkthrough', root.querySelectorAll('.method-step').length === 6);
    check(id + ' retrieval matches released capability', root.querySelector('.method-submit').disabled === !methods.includes(id));
    check(id + ' answer stays disabled on current backend', root.querySelector('.method-kind option[value="answer"]').disabled);
  }
  check('capability and page inspection posts no job', __ownedPreview.posts.length === postsBefore);
  return { kind: 'LOCAL_BROWSER_CURRENT_PRODUCTION_CAPABILITY_FIXTURE', checks, passed: checks.length,
    released_retrieval_methods: methods, answer_methods: [], auth_and_api_simulated: true, aws_calls: 0, provider_calls: 0 };
}
