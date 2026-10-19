import {beginLogin,finishLogin,validateConfig} from "./auth.mjs";

const get=id=>document.getElementById(id);
try {
  const callback=location.pathname==="/auth/callback"&&location.search?location.href:null;
  if(callback)history.replaceState(null,"","/auth/callback");
  const response=await fetch("/config.json",{cache:"no-store",credentials:"omit",redirect:"error"});
  if(!response.ok)throw new Error("CONFIG_UNAVAILABLE");
  const config=validateConfig(await response.json(),location.origin);
  get("login").disabled=false;
  get("login").addEventListener("click",async()=>{
    get("login").disabled=true;
    try{location.assign(await beginLogin(config,sessionStorage));}
    catch{get("status").textContent="Could not start sign-in. Please try again.";get("login").disabled=false;}
  });
  get("logout").addEventListener("click",()=>{
    sessionStorage.removeItem("kg.oauth");
    const url=new URL("/logout",config.auth_domain);
    url.search=new URLSearchParams({client_id:config.client_id,logout_uri:config.logout_url});location.assign(url.href);
  });
  if(callback){
    const token=await finishLogin(config,callback,sessionStorage);
    get("logout").hidden=false;
    const result=await fetch("/v1/status",{headers:{Authorization:`Bearer ${token}`},cache:"no-store",credentials:"omit",redirect:"error"});
    if(!result.ok)throw new Error("ACCESS_UNAVAILABLE");
    get("status").textContent=(await result.json()).message;get("login").hidden=true;
  }else get("status").textContent="Ready. Sign in with your existing reviewer account.";
}catch{get("status").textContent="Sign-in could not be completed. Start sign-in again.";}
