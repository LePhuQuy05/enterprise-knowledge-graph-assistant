async () => {
const checks=[]; const check=(name,pass)=>{checks.push({name,pass:!!pass});if(!pass)throw new Error(name);};
const wait=async fn=>{for(let i=0;i<80;i++){if(fn())return;await new Promise(r=>setTimeout(r,100));}throw new Error("WAIT_TIMEOUT");};
await wait(()=>globalThis.__answerPreview); document.querySelector('dialog[open]')?.close();
const ids=['bm25','vector','hybrid','hybrid_reranked','hybrid_multi_query','hybrid_multi_query_balanced','graph_1hop','graph_2hop'];
const root=id=>document.getElementById('page-method-'+id);
for(const id of ids){
document.querySelector('[data-page="method-'+id+'"]').click();await wait(()=>root(id).querySelectorAll('.method-step').length===6);
check(id+' is a separate navigable page',!root(id).hidden&&document.querySelectorAll('[data-view]:not([hidden])').length===1);
check(id+' uses collection316',root(id).querySelector('.method-record-meta').textContent.includes('316 documents'));
for(const button of root(id).querySelectorAll('.method-step'))button.click();
check(id+' has source verification step',root(id).querySelector('.method-detail').textContent.includes('Citation'));
if(!id.startsWith('graph_')){
root(id).querySelectorAll('.method-step')[4].click();
const citation=root(id).querySelector('.method-claim button');check(id+' has recorded answer citations',citation);
citation.click();check(id+' opens original passage',root(id).querySelector('.method-source blockquote')?.textContent.length>5);
}
}
check('method navigation opens parent menu',document.getElementById('methods-navigation').open);
check('opening examples posts no jobs',__ownedPreview.posts.length===0);
let r=root('graph_2hop');check('graph uses collection316 only',r.querySelector('.method-graph-counts').textContent.includes('5/316'));
check('graph has real edges',r.querySelectorAll('.graph-edge').length>0);
const edge=r.querySelector('.graph-edge');edge.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
check('keyboard opens assertion proof',r.querySelector('.method-graph-inspector blockquote')?.textContent.length>5);
r.querySelector('.method-graph-inspector button').click();
check('graph source quote is readable',r.querySelector('.method-source blockquote')?.textContent.length>5);
const all=r.querySelector('.method-graph-all');all.checked=true;all.dispatchEvent(new Event('change',{bubbles:true}));
check('all assertions can be inspected',r.querySelectorAll('.graph-edge').length===13);
document.querySelector('[data-page="method-hybrid"]').click();r=root('hybrid');
r.querySelector('textarea').value=__answerPreview.question;
r.querySelector('.method-kind').value='retrieval';r.querySelector('.method-form').requestSubmit();
await wait(()=>r.querySelector('.method-record-kind').textContent.includes('YOUR RUN'));
check('new question stays in its method page',!r.hidden);
check('one retrieval POST for one submit',__ownedPreview.posts.length===1);
r.querySelectorAll('.method-step')[3].click();check('new retrieval shows actual eight context chunks',r.querySelectorAll('.method-detail .method-ranks li').length===8);
r.querySelectorAll('.method-step')[4].click();check('retrieval does not pretend to generate',r.querySelector('.method-detail').textContent.includes('chỉ retrieval'));
r.querySelector('.method-kind').value='answer';r.querySelector('.method-form').requestSubmit();
await wait(()=>r.querySelector('.method-status').textContent.includes('COMPLETED')&&r.querySelectorAll('.method-step').length===6);
r.querySelectorAll('.method-step')[4].click();
check('new answer renders four recorded claims',r.querySelectorAll('.method-claim').length===4);
check('answer is exactly second explicit POST',__ownedPreview.posts.length===2);
r.querySelector('.method-claim button').click();
await wait(()=>r.querySelector('.method-source blockquote')?.textContent.length>5);
check('owned source returns into originating method page',!r.hidden&&r.querySelector('.method-source').textContent.includes('Selected evidence'));
const before=__ownedPreview.posts.length;r.querySelector('.method-refresh').click();await new Promise(r=>setTimeout(r,250));
check('refresh does not resubmit',__ownedPreview.posts.length===before);
check('legacy query POST is unused',!__preview.calls.some(c=>c.path==='/v1/queries'&&c.method==='POST'));
document.getElementById('logout').click();
check('logout clears all method questions',ids.every(id=>root(id).querySelector('textarea').value===''));
check('logout clears private passages',ids.every(id=>!root(id).querySelector('.method-source').textContent.trim()));
check('logout disables all run buttons',ids.every(id=>root(id).querySelector('.method-submit').disabled));
return {kind:'BROWSER_METHOD_PAGES_RECORDED_DATA_SIMULATED_AUTH_QUEUE_PROVIDER',checks,passed:checks.filter(c=>c.pass).length,aws_calls:0,provider_calls:0,viewport:innerWidth};
}
