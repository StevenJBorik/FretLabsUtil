var n,l$2,u$1,t$1,o$2,r$1,f$1,e$1,c$3={},s$3=[],a$2=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i,v$2=Array.isArray;function h$2(n,l){for(var u in l)n[u]=l[u];return n}function p$2(n){var l=n.parentNode;l&&l.removeChild(n);}function y$1(l,u,i){var t,o,r,f={};for(r in u)"key"==r?t=u[r]:"ref"==r?o=u[r]:f[r]=u[r];if(arguments.length>2&&(f.children=arguments.length>3?n.call(arguments,2):i),"function"==typeof l&&null!=l.defaultProps)for(r in l.defaultProps)void 0===f[r]&&(f[r]=l.defaultProps[r]);return d$2(l,f,t,o,null)}function d$2(n,i,t,o,r){var f={type:n,props:i,key:t,ref:o,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,__h:null,constructor:void 0,__v:null==r?++u$1:r};return null==r&&null!=l$2.vnode&&l$2.vnode(f),f}function _$1(){return {current:null}}function k$2(n){return n.children}function b$2(n,l){this.props=n,this.context=l;}function g$2(n,l){if(null==l)return n.__?g$2(n.__,n.__.__k.indexOf(n)+1):null;for(var u;l<n.__k.length;l++)if(null!=(u=n.__k[l])&&null!=u.__e)return u.__e;return "function"==typeof n.type?g$2(n):null}function m$2(n){var l,u;if(null!=(n=n.__)&&null!=n.__c){for(n.__e=n.__c.base=null,l=0;l<n.__k.length;l++)if(null!=(u=n.__k[l])&&null!=u.__e){n.__e=n.__c.base=u.__e;break}return m$2(n)}}function w$2(n){(!n.__d&&(n.__d=!0)&&t$1.push(n)&&!x.__r++||o$2!==l$2.debounceRendering)&&((o$2=l$2.debounceRendering)||r$1)(x);}function x(){var n,l,u,i,o,r,e,c;for(t$1.sort(f$1);n=t$1.shift();)n.__d&&(l=t$1.length,i=void 0,o=void 0,e=(r=(u=n).__v).__e,(c=u.__P)&&(i=[],(o=h$2({},r)).__v=r.__v+1,L(c,r,o,u.__n,void 0!==c.ownerSVGElement,null!=r.__h?[e]:null,i,null==e?g$2(r):e,r.__h),M$1(i,r),r.__e!=e&&m$2(r)),t$1.length>l&&t$1.sort(f$1));x.__r=0;}function P(n,l,u,i,t,o,r,f,e,a){var h,p,y,_,b,m,w,x=i&&i.__k||s$3,P=x.length;for(u.__k=[],h=0;h<l.length;h++)if(null!=(_=u.__k[h]=null==(_=l[h])||"boolean"==typeof _||"function"==typeof _?null:"string"==typeof _||"number"==typeof _||"bigint"==typeof _?d$2(null,_,null,null,_):v$2(_)?d$2(k$2,{children:_},null,null,null):_.__b>0?d$2(_.type,_.props,_.key,_.ref?_.ref:null,_.__v):_)){if(_.__=u,_.__b=u.__b+1,null===(y=x[h])||y&&_.key==y.key&&_.type===y.type)x[h]=void 0;else for(p=0;p<P;p++){if((y=x[p])&&_.key==y.key&&_.type===y.type){x[p]=void 0;break}y=null;}L(n,_,y=y||c$3,t,o,r,f,e,a),b=_.__e,(p=_.ref)&&y.ref!=p&&(w||(w=[]),y.ref&&w.push(y.ref,null,_),w.push(p,_.__c||b,_)),null!=b?(null==m&&(m=b),"function"==typeof _.type&&_.__k===y.__k?_.__d=e=C$1(_,e,n):e=$$1(n,_,y,x,b,e),"function"==typeof u.type&&(u.__d=e)):e&&y.__e==e&&e.parentNode!=n&&(e=g$2(y));}for(u.__e=m,h=P;h--;)null!=x[h]&&("function"==typeof u.type&&null!=x[h].__e&&x[h].__e==u.__d&&(u.__d=A(i).nextSibling),q$1(x[h],x[h]));if(w)for(h=0;h<w.length;h++)O(w[h],w[++h],w[++h]);}function C$1(n,l,u){for(var i,t=n.__k,o=0;t&&o<t.length;o++)(i=t[o])&&(i.__=n,l="function"==typeof i.type?C$1(i,l,u):$$1(u,i,i,t,i.__e,l));return l}function S(n,l){return l=l||[],null==n||"boolean"==typeof n||(v$2(n)?n.some(function(n){S(n,l);}):l.push(n)),l}function $$1(n,l,u,i,t,o){var r,f,e;if(void 0!==l.__d)r=l.__d,l.__d=void 0;else if(null==u||t!=o||null==t.parentNode)n:if(null==o||o.parentNode!==n)n.appendChild(t),r=null;else {for(f=o,e=0;(f=f.nextSibling)&&e<i.length;e+=1)if(f==t)break n;n.insertBefore(t,o),r=o;}return void 0!==r?r:t.nextSibling}function A(n){var l,u,i;if(null==n.type||"string"==typeof n.type)return n.__e;if(n.__k)for(l=n.__k.length-1;l>=0;l--)if((u=n.__k[l])&&(i=A(u)))return i;return null}function H(n,l,u,i,t){var o;for(o in u)"children"===o||"key"===o||o in l||T(n,o,null,u[o],i);for(o in l)t&&"function"!=typeof l[o]||"children"===o||"key"===o||"value"===o||"checked"===o||u[o]===l[o]||T(n,o,l[o],u[o],i);}function I$1(n,l,u){"-"===l[0]?n.setProperty(l,null==u?"":u):n[l]=null==u?"":"number"!=typeof u||a$2.test(l)?u:u+"px";}function T(n,l,u,i,t){var o;n:if("style"===l)if("string"==typeof u)n.style.cssText=u;else {if("string"==typeof i&&(n.style.cssText=i=""),i)for(l in i)u&&l in u||I$1(n.style,l,"");if(u)for(l in u)i&&u[l]===i[l]||I$1(n.style,l,u[l]);}else if("o"===l[0]&&"n"===l[1])o=l!==(l=l.replace(/Capture$/,"")),l=l.toLowerCase()in n?l.toLowerCase().slice(2):l.slice(2),n.l||(n.l={}),n.l[l+o]=u,u?i||n.addEventListener(l,o?z$1:j$1,o):n.removeEventListener(l,o?z$1:j$1,o);else if("dangerouslySetInnerHTML"!==l){if(t)l=l.replace(/xlink(H|:h)/,"h").replace(/sName$/,"s");else if("width"!==l&&"height"!==l&&"href"!==l&&"list"!==l&&"form"!==l&&"tabIndex"!==l&&"download"!==l&&"rowSpan"!==l&&"colSpan"!==l&&l in n)try{n[l]=null==u?"":u;break n}catch(n){}"function"==typeof u||(null==u||!1===u&&"-"!==l[4]?n.removeAttribute(l):n.setAttribute(l,u));}}function j$1(n){return this.l[n.type+!1](l$2.event?l$2.event(n):n)}function z$1(n){return this.l[n.type+!0](l$2.event?l$2.event(n):n)}function L(n,u,i,t,o,r,f,e,c){var s,a,p,y,d,_,g,m,w,x,C,S,$,A,H,I=u.type;if(void 0!==u.constructor)return null;null!=i.__h&&(c=i.__h,e=u.__e=i.__e,u.__h=null,r=[e]),(s=l$2.__b)&&s(u);try{n:if("function"==typeof I){if(m=u.props,w=(s=I.contextType)&&t[s.__c],x=s?w?w.props.value:s.__:t,i.__c?g=(a=u.__c=i.__c).__=a.__E:("prototype"in I&&I.prototype.render?u.__c=a=new I(m,x):(u.__c=a=new b$2(m,x),a.constructor=I,a.render=B$1),w&&w.sub(a),a.props=m,a.state||(a.state={}),a.context=x,a.__n=t,p=a.__d=!0,a.__h=[],a._sb=[]),null==a.__s&&(a.__s=a.state),null!=I.getDerivedStateFromProps&&(a.__s==a.state&&(a.__s=h$2({},a.__s)),h$2(a.__s,I.getDerivedStateFromProps(m,a.__s))),y=a.props,d=a.state,a.__v=u,p)null==I.getDerivedStateFromProps&&null!=a.componentWillMount&&a.componentWillMount(),null!=a.componentDidMount&&a.__h.push(a.componentDidMount);else {if(null==I.getDerivedStateFromProps&&m!==y&&null!=a.componentWillReceiveProps&&a.componentWillReceiveProps(m,x),!a.__e&&null!=a.shouldComponentUpdate&&!1===a.shouldComponentUpdate(m,a.__s,x)||u.__v===i.__v){for(u.__v!==i.__v&&(a.props=m,a.state=a.__s,a.__d=!1),a.__e=!1,u.__e=i.__e,u.__k=i.__k,u.__k.forEach(function(n){n&&(n.__=u);}),C=0;C<a._sb.length;C++)a.__h.push(a._sb[C]);a._sb=[],a.__h.length&&f.push(a);break n}null!=a.componentWillUpdate&&a.componentWillUpdate(m,a.__s,x),null!=a.componentDidUpdate&&a.__h.push(function(){a.componentDidUpdate(y,d,_);});}if(a.context=x,a.props=m,a.__P=n,S=l$2.__r,$=0,"prototype"in I&&I.prototype.render){for(a.state=a.__s,a.__d=!1,S&&S(u),s=a.render(a.props,a.state,a.context),A=0;A<a._sb.length;A++)a.__h.push(a._sb[A]);a._sb=[];}else do{a.__d=!1,S&&S(u),s=a.render(a.props,a.state,a.context),a.state=a.__s;}while(a.__d&&++$<25);a.state=a.__s,null!=a.getChildContext&&(t=h$2(h$2({},t),a.getChildContext())),p||null==a.getSnapshotBeforeUpdate||(_=a.getSnapshotBeforeUpdate(y,d)),P(n,v$2(H=null!=s&&s.type===k$2&&null==s.key?s.props.children:s)?H:[H],u,i,t,o,r,f,e,c),a.base=u.__e,u.__h=null,a.__h.length&&f.push(a),g&&(a.__E=a.__=null),a.__e=!1;}else null==r&&u.__v===i.__v?(u.__k=i.__k,u.__e=i.__e):u.__e=N(i.__e,u,i,t,o,r,f,c);(s=l$2.diffed)&&s(u);}catch(n){u.__v=null,(c||null!=r)&&(u.__e=e,u.__h=!!c,r[r.indexOf(e)]=null),l$2.__e(n,u,i);}}function M$1(n,u){l$2.__c&&l$2.__c(u,n),n.some(function(u){try{n=u.__h,u.__h=[],n.some(function(n){n.call(u);});}catch(n){l$2.__e(n,u.__v);}});}function N(l,u,i,t,o,r,f,e){var s,a,h,y=i.props,d=u.props,_=u.type,k=0;if("svg"===_&&(o=!0),null!=r)for(;k<r.length;k++)if((s=r[k])&&"setAttribute"in s==!!_&&(_?s.localName===_:3===s.nodeType)){l=s,r[k]=null;break}if(null==l){if(null===_)return document.createTextNode(d);l=o?document.createElementNS("http://www.w3.org/2000/svg",_):document.createElement(_,d.is&&d),r=null,e=!1;}if(null===_)y===d||e&&l.data===d||(l.data=d);else {if(r=r&&n.call(l.childNodes),a=(y=i.props||c$3).dangerouslySetInnerHTML,h=d.dangerouslySetInnerHTML,!e){if(null!=r)for(y={},k=0;k<l.attributes.length;k++)y[l.attributes[k].name]=l.attributes[k].value;(h||a)&&(h&&(a&&h.__html==a.__html||h.__html===l.innerHTML)||(l.innerHTML=h&&h.__html||""));}if(H(l,d,y,o,e),h)u.__k=[];else if(P(l,v$2(k=u.props.children)?k:[k],u,i,t,o&&"foreignObject"!==_,r,f,r?r[0]:i.__k&&g$2(i,0),e),null!=r)for(k=r.length;k--;)null!=r[k]&&p$2(r[k]);e||("value"in d&&void 0!==(k=d.value)&&(k!==l.value||"progress"===_&&!k||"option"===_&&k!==y.value)&&T(l,"value",k,y.value,!1),"checked"in d&&void 0!==(k=d.checked)&&k!==l.checked&&T(l,"checked",k,y.checked,!1));}return l}function O(n,u,i){try{"function"==typeof n?n(u):n.current=u;}catch(n){l$2.__e(n,i);}}function q$1(n,u,i){var t,o;if(l$2.unmount&&l$2.unmount(n),(t=n.ref)&&(t.current&&t.current!==n.__e||O(t,null,u)),null!=(t=n.__c)){if(t.componentWillUnmount)try{t.componentWillUnmount();}catch(n){l$2.__e(n,u);}t.base=t.__P=null,n.__c=void 0;}if(t=n.__k)for(o=0;o<t.length;o++)t[o]&&q$1(t[o],u,i||"function"!=typeof n.type);i||null==n.__e||p$2(n.__e),n.__=n.__e=n.__d=void 0;}function B$1(n,l,u){return this.constructor(n,u)}function D$1(u,i,t){var o,r,f;l$2.__&&l$2.__(u,i),r=(o="function"==typeof t)?null:t&&t.__k||i.__k,f=[],L(i,u=(!o&&t||i).__k=y$1(k$2,null,[u]),r||c$3,c$3,void 0!==i.ownerSVGElement,!o&&t?[t]:r?null:i.firstChild?n.call(i.childNodes):null,f,!o&&t?t:r?r.__e:i.firstChild,o),M$1(f,u);}function F(l,u,i){var t,o,r,f,e=h$2({},l.props);for(r in l.type&&l.type.defaultProps&&(f=l.type.defaultProps),u)"key"==r?t=u[r]:"ref"==r?o=u[r]:e[r]=void 0===u[r]&&void 0!==f?f[r]:u[r];return arguments.length>2&&(e.children=arguments.length>3?n.call(arguments,2):i),d$2(l.type,e,t||l.key,o||l.ref,null)}function G(n,l){var u={__c:l="__cC"+e$1++,__:n,Consumer:function(n,l){return n.children(l)},Provider:function(n){var u,i;return this.getChildContext||(u=[],(i={})[l]=this,this.getChildContext=function(){return i},this.shouldComponentUpdate=function(n){this.props.value!==n.value&&u.some(function(n){n.__e=!0,w$2(n);});},this.sub=function(n){u.push(n);var l=n.componentWillUnmount;n.componentWillUnmount=function(){u.splice(u.indexOf(n),1),l&&l.call(n);};}),n.children}};return u.Provider.__=u.Consumer.contextType=u}n=s$3.slice,l$2={__e:function(n,l,u,i){for(var t,o,r;l=l.__;)if((t=l.__c)&&!t.__)try{if((o=t.constructor)&&null!=o.getDerivedStateFromError&&(t.setState(o.getDerivedStateFromError(n)),r=t.__d),null!=t.componentDidCatch&&(t.componentDidCatch(n,i||{}),r=t.__d),r)return t.__E=t}catch(l){n=l;}throw n}},u$1=0,b$2.prototype.setState=function(n,l){var u;u=null!=this.__s&&this.__s!==this.state?this.__s:this.__s=h$2({},this.state),"function"==typeof n&&(n=n(h$2({},u),this.props)),n&&h$2(u,n),null!=n&&this.__v&&(l&&this._sb.push(l),w$2(this));},b$2.prototype.forceUpdate=function(n){this.__v&&(this.__e=!0,n&&this.__h.push(n),w$2(this));},b$2.prototype.render=k$2,t$1=[],r$1="function"==typeof Promise?Promise.prototype.then.bind(Promise.resolve()):setTimeout,f$1=function(n,l){return n.__v.__b-l.__v.__b},x.__r=0,e$1=0;

var _=0;function o$1(o,e,n,t,f,l){var s,u,a={};for(u in e)"ref"==u?s=e[u]:a[u]=e[u];var i={type:o,props:a,key:n,ref:s,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,__h:null,constructor:void 0,__v:--_,__source:f,__self:l};if("function"==typeof o&&(s=o.defaultProps))for(u in s)void 0===a[u]&&(a[u]=s[u]);return l$2.vnode&&l$2.vnode(i),i}

/******************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    if (typeof b !== "function" && b !== null)
        throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

var __assign = function() {
    __assign = Object.assign || function __assign(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p)) t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

var t,r,u,i,o=0,f=[],c$2=[],e=l$2.__b,a$1=l$2.__r,v$1=l$2.diffed,l$1=l$2.__c,m$1=l$2.unmount;function d$1(t,u){l$2.__h&&l$2.__h(r,t,o||u),o=0;var i=r.__H||(r.__H={__:[],__h:[]});return t>=i.__.length&&i.__.push({__V:c$2}),i.__[t]}function h$1(n){return o=1,s$2(B,n)}function s$2(n,u,i){var o=d$1(t++,2);if(o.t=n,!o.__c&&(o.__=[i?i(u):B(void 0,u),function(n){var t=o.__N?o.__N[0]:o.__[0],r=o.t(t,n);t!==r&&(o.__N=[r,o.__[1]],o.__c.setState({}));}],o.__c=r,!r.u)){var f=function(n,t,r){if(!o.__c.__H)return !0;var u=o.__c.__H.__.filter(function(n){return n.__c});if(u.every(function(n){return !n.__N}))return !c||c.call(this,n,t,r);var i=!1;return u.forEach(function(n){if(n.__N){var t=n.__[0];n.__=n.__N,n.__N=void 0,t!==n.__[0]&&(i=!0);}}),!(!i&&o.__c.props===n)&&(!c||c.call(this,n,t,r))};r.u=!0;var c=r.shouldComponentUpdate,e=r.componentWillUpdate;r.componentWillUpdate=function(n,t,r){if(this.__e){var u=c;c=void 0,f(n,t,r),c=u;}e&&e.call(this,n,t,r);},r.shouldComponentUpdate=f;}return o.__N||o.__}function p$1(u,i){var o=d$1(t++,3);!l$2.__s&&z(o.__H,i)&&(o.__=u,o.i=i,r.__H.__h.push(o));}function q(n){var u=r.context[n.__c],i=d$1(t++,9);return i.c=n,u?(null==i.__&&(i.__=!0,u.sub(r)),u.props.value):n.__}function b$1(){for(var t;t=f.shift();)if(t.__P&&t.__H)try{t.__H.__h.forEach(k$1),t.__H.__h.forEach(w$1),t.__H.__h=[];}catch(r){t.__H.__h=[],l$2.__e(r,t.__v);}}l$2.__b=function(n){r=null,e&&e(n);},l$2.__r=function(n){a$1&&a$1(n),t=0;var i=(r=n.__c).__H;i&&(u===r?(i.__h=[],r.__h=[],i.__.forEach(function(n){n.__N&&(n.__=n.__N),n.__V=c$2,n.__N=n.i=void 0;})):(i.__h.forEach(k$1),i.__h.forEach(w$1),i.__h=[],t=0)),u=r;},l$2.diffed=function(t){v$1&&v$1(t);var o=t.__c;o&&o.__H&&(o.__H.__h.length&&(1!==f.push(o)&&i===l$2.requestAnimationFrame||((i=l$2.requestAnimationFrame)||j)(b$1)),o.__H.__.forEach(function(n){n.i&&(n.__H=n.i),n.__V!==c$2&&(n.__=n.__V),n.i=void 0,n.__V=c$2;})),u=r=null;},l$2.__c=function(t,r){r.some(function(t){try{t.__h.forEach(k$1),t.__h=t.__h.filter(function(n){return !n.__||w$1(n)});}catch(u){r.some(function(n){n.__h&&(n.__h=[]);}),r=[],l$2.__e(u,t.__v);}}),l$1&&l$1(t,r);},l$2.unmount=function(t){m$1&&m$1(t);var r,u=t.__c;u&&u.__H&&(u.__H.__.forEach(function(n){try{k$1(n);}catch(n){r=n;}}),u.__H=void 0,r&&l$2.__e(r,u.__v));};var g$1="function"==typeof requestAnimationFrame;function j(n){var t,r=function(){clearTimeout(u),g$1&&cancelAnimationFrame(t),setTimeout(n);},u=setTimeout(r,100);g$1&&(t=requestAnimationFrame(r));}function k$1(n){var t=r,u=n.__c;"function"==typeof u&&(n.__c=void 0,u()),r=t;}function w$1(n){var t=r;n.__c=n.__(),r=t;}function z(n,t){return !n||n.length!==t.length||t.some(function(t,r){return t!==n[r]})}function B(n,t){return "function"==typeof t?t(n):t}

var a={};function c$1(n,t){for(var r in t)n[r]=t[r];return n}function s$1(n,t,r){var i,o=/(?:\?([^#]*))?(#.*)?$/,e=n.match(o),u={};if(e&&e[1])for(var f=e[1].split("&"),c=0;c<f.length;c++){var s=f[c].split("=");u[decodeURIComponent(s[0])]=decodeURIComponent(s.slice(1).join("="));}n=d(n.replace(o,"")),t=d(t||"");for(var h=Math.max(n.length,t.length),v=0;v<h;v++)if(t[v]&&":"===t[v].charAt(0)){var l=t[v].replace(/(^:|[+*?]+$)/g,""),p=(t[v].match(/[+*?]+$/)||a)[0]||"",m=~p.indexOf("+"),y=~p.indexOf("*"),U=n[v]||"";if(!U&&!y&&(p.indexOf("?")<0||m)){i=!1;break}if(u[l]=decodeURIComponent(U),m||y){u[l]=n.slice(v).map(decodeURIComponent).join("/");break}}else if(t[v]!==n[v]){i=!1;break}return (!0===r.default||!1!==i)&&u}function h(n,t){return n.rank<t.rank?1:n.rank>t.rank?-1:n.index-t.index}function v(n,t){return n.index=t,n.rank=function(n){return n.props.default?0:d(n.props.path).map(l).join("")}(n),n.props}function d(n){return n.replace(/(^\/+|\/+$)/g,"").split("/")}function l(n){return ":"==n.charAt(0)?1+"*+?".indexOf(n.charAt(n.length-1))||4:5}var p={},m=[],y=[],U=null,g={url:R()},k=G(g);function C(){var n=q(k);if(n===g){var t=h$1()[1];p$1(function(){return y.push(t),function(){return y.splice(y.indexOf(t),1)}},[]);}return [n,$]}function R(){var n;return ""+((n=U&&U.location?U.location:U&&U.getCurrentLocation?U.getCurrentLocation():"undefined"!=typeof location?location:p).pathname||"")+(n.search||"")}function $(n,t){return void 0===t&&(t=!1),"string"!=typeof n&&n.url&&(t=n.replace,n=n.url),function(n){for(var t=m.length;t--;)if(m[t].canRoute(n))return !0;return !1}(n)&&function(n,t){void 0===t&&(t="push"),U&&U[t]?U[t](n):"undefined"!=typeof history&&history[t+"State"]&&history[t+"State"](null,null,n);}(n,t?"replace":"push"),I(n)}function I(n){for(var t=!1,r=0;r<m.length;r++)m[r].routeTo(n)&&(t=!0);return t}function M(n){if(n&&n.getAttribute){var t=n.getAttribute("href"),r=n.getAttribute("target");if(t&&t.match(/^\//g)&&(!r||r.match(/^_?self$/i)))return $(t)}}function b(n){return n.stopImmediatePropagation&&n.stopImmediatePropagation(),n.stopPropagation&&n.stopPropagation(),n.preventDefault(),!1}function W(n){if(!(n.ctrlKey||n.metaKey||n.altKey||n.shiftKey||n.button)){var t=n.target;do{if("a"===t.localName&&t.getAttribute("href")){if(t.hasAttribute("data-native")||t.hasAttribute("native"))return;if(M(t))return b(n)}}while(t=t.parentNode)}}var w=!1;function D(n){n.history&&(U=n.history),this.state={url:n.url||R()};}c$1(D.prototype=new b$2,{shouldComponentUpdate:function(n){return !0!==n.static||n.url!==this.props.url||n.onChange!==this.props.onChange},canRoute:function(n){var t=S(this.props.children);return void 0!==this.g(t,n)},routeTo:function(n){this.setState({url:n});var t=this.canRoute(n);return this.p||this.forceUpdate(),t},componentWillMount:function(){this.p=!0;},componentDidMount:function(){var n=this;w||(w=!0,U||addEventListener("popstate",function(){I(R());}),addEventListener("click",W)),m.push(this),U&&(this.u=U.listen(function(t){var r=t.location||t;n.routeTo(""+(r.pathname||"")+(r.search||""));})),this.p=!1;},componentWillUnmount:function(){"function"==typeof this.u&&this.u(),m.splice(m.indexOf(this),1);},componentWillUpdate:function(){this.p=!0;},componentDidUpdate:function(){this.p=!1;},g:function(n,t){n=n.filter(v).sort(h);for(var r=0;r<n.length;r++){var i=n[r],o=s$1(t,i.props.path,i.props);if(o)return [i,o]}},render:function(n,t){var e,u,f=n.onChange,a=t.url,s=this.c,h=this.g(S(n.children),a);if(h&&(u=F(h[0],c$1(c$1({url:a,matches:e=h[1]},e),{key:void 0,ref:void 0}))),a!==(s&&s.url)){c$1(g,s=this.c={url:a,previous:s&&s.url,current:u,path:u?u.props.path:null,matches:e}),s.router=this,s.active=u?[u]:[];for(var v=y.length;v--;)y[v]({});"function"==typeof f&&f(s);}return y$1(k.Provider,{value:s},u)}});var E=function(n){return y$1("a",c$1({onClick:W},n))};

var s=["className","activeClass","activeClassName","path"];function c(l){var c=l.className,n=l.activeClass,u=l.activeClassName,i=l.path,p=function(a,t){if(null==a)return {};var r,e,s={},l=Object.keys(a);for(e=0;e<l.length;e++)t.indexOf(r=l[e])>=0||(s[r]=a[r]);return s}(l,s),h=C()[0],f=i&&h.path&&s$1(h.path,i,{})||s$1(h.url,p.href,{}),o=p.class||c||"",m=f&&(n||u)||"";return p.class=o+(o&&m&&" ")+m,y$1(E,p)}

var App$1 = /** @class */ (function (_super) {
    __extends(App, _super);
    function App() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.closeNav = function () {
            var navOpen = _this.state.navOpen;
            if (navOpen === true) {
                _this.setState({ navOpen: false });
            }
        };
        return _this;
    }
    App.prototype.render = function (_, _a) {
        var _this = this;
        var navOpen = _a.navOpen;
        return (o$1("nav", __assign({ "class": ['navigation-wrapper', navOpen ? 'navigation-open' : ''].join(' ') }, { children: [o$1("button", __assign({ onClick: function () { return _this.setState({ navOpen: true }); } }, { children: "\u2630" })), o$1("div", __assign({ "class": "links-container" }, { children: [o$1(c, __assign({ href: "/live", activeClassName: "active", onClick: this.closeNav }, { children: "Live Detection" })), o$1(c, __assign({ href: "/file", activeClassName: "active", onClick: this.closeNav }, { children: "File Analysis" })), o$1(c, __assign({ href: "/settings", activeClassName: "active", onClick: this.closeNav }, { children: "Settings" })), o$1(c, __assign({ href: "/about", activeClassName: "active", onClick: this.closeNav }, { children: "About" }))] }))] })));
    };
    return App;
}(b$2));

var recorderWorkletURL = "recorderWorkletProcessor-f0ba7422.js";

function requestUserMedia() {
    return navigator.mediaDevices.getUserMedia({ audio: true, video: false });
}
function createAudioContext() {
    var AudioContext = window.AudioContext || window.webkitAudioContext;
    return new AudioContext();
}
function createAudioSource(audioContext, stream) {
    return audioContext.createMediaStreamSource(stream);
}
function getSourceMeta(audioSource) {
    return {
        numberOfChannels: audioSource.channelCount,
        sampleRate: audioSource.context.sampleRate
    };
}
function createRecordingDevice(audioContext) {
    return new Promise(function (resolve, reject) {
        audioContext.audioWorklet
            .addModule(recorderWorkletURL)
            .then(function () {
            var recorder = new AudioWorkletNode(audioContext, 'recorder-worklet');
            resolve(recorder);
        })["catch"](function (e) {
            reject(e);
        });
    });
}
function createAnalyserDevice(audioContext) {
    var analyzer = audioContext.createAnalyser();
    analyzer.fftSize = 2048;
    return analyzer;
}
function createDataArrayForAnalyzerDevice(analyzer) {
    var bufferLength = analyzer.frequencyBinCount;
    var dataArray = new Uint8Array(bufferLength);
    analyzer.getByteTimeDomainData(dataArray);
    return dataArray;
}
function connectAudioNodes(audioSource, audioRecorder) {
    audioSource.connect(audioRecorder);
}

var keyFinderWorkerURL = "keyFinderProgressiveWorker-417e9013.js";

function initializeKeyFinder({ sampleRate, numberOfChannels }) {
  const worker = new Worker(keyFinderWorkerURL);
  worker.postMessage({
    funcName: 'initialize',
    data: [sampleRate, numberOfChannels],
  });
  return worker;
}

function extractResultFromByteArray(byteArray) {
  return byteArray.reduce(
    (acc, cur) => `${acc}${String.fromCharCode(cur)}`,
    ''
  );
}

function zipChannelsAtOffset(
  channelData,
  offset,
  sampleRate,
  numberOfChannels
) {
  const segment = new Float32Array(sampleRate * numberOfChannels);
  for (let i = 0; i < sampleRate; i += 1) {
    for (let j = 0; j < numberOfChannels; j += 1) {
      segment[i + j] = channelData[j][offset + i];
    }
  }
  return segment;
}

var customSettings = JSON.parse(localStorage.getItem('customSettings'));
var majorKeys = [
    'C Major',
    'G Major',
    'D Major',
    'A Major',
    'E Major',
    'B Major',
    'Gb Major',
    'Db Major',
    'Ab Major',
    'Eb Major',
    'Bb Major',
    'F Major',
];
var minorKeys = [
    'A Minor',
    'E Minor',
    'B Minor',
    'Gb Minor',
    'Db Minor',
    'Ab Minor',
    'Eb Minor',
    'Bb Minor',
    'F Minor',
    'C Minor',
    'G Minor',
    'D Minor',
];
var defaultKeysNotation = {
    'C Major': 'C',
    'G Major': 'G',
    'D Major': 'D',
    'A Major': 'A',
    'E Major': 'E',
    'B Major': 'B',
    'Gb Major': 'G♭',
    'Db Major': 'D♭',
    'Ab Major': 'A♭',
    'Eb Major': 'E♭',
    'Bb Major': 'B♭',
    'F Major': 'F',
    'A Minor': 'Am',
    'E Minor': 'Em',
    'B Minor': 'Bm',
    'Gb Minor': 'G♭m',
    'Db Minor': 'D♭m',
    'Ab Minor': 'A♭m',
    'Eb Minor': 'E♭m',
    'Bb Minor': 'B♭m',
    'F Minor': 'Fm',
    'C Minor': 'Cm',
    'G Minor': 'Gm',
    'D Minor': 'Dm'
};
var keysNotation = customSettings && customSettings.keysNotation
    ? customSettings.keysNotation
    : defaultKeysNotation;
var defaultTheme = 'light';
var theme$1 = customSettings && customSettings.theme ? customSettings.theme : defaultTheme;
var defaultKeyAtTopPosition = 'C Major';
var keyAtTopPosition = customSettings && customSettings.keyAtTopPosition
    ? customSettings.keyAtTopPosition
    : defaultKeyAtTopPosition;
var maxNumberOfThreads = navigator.hardwareConcurrency;
var defaultNumberOfThreads = navigator.hardwareConcurrency - 1;
var numberOfThreads = customSettings && customSettings.numberOfThreads
    ? customSettings.numberOfThreads
    : defaultNumberOfThreads || 1;

var lightThemeColors = {
    '--foreground-color': ' #24292E',
    '--background-color': '#FFFFFF',
    '--gray-color': '#C0C1C1',
    '--primary-color': '#3778C2',
    '--primary-darker-color': '#28559A',
    '--secondary-color': '#FF6801',
    '--danger-color': '#CC0000'
};
var darkThemeColors = {
    '--foreground-color': '#C9D1D9',
    '--background-color': '#0D1117',
    '--gray-color': '#454444',
    '--primary-color': '#4B9FE1',
    '--primary-darker-color': '#63BCE5',
    '--secondary-color': '#FF6801',
    '--danger-color': '#CC0000'
};
function updateColors(colors) {
    return Object.keys(colors).reduce(function (acc, cur) {
        document.documentElement.style.setProperty(cur, colors[cur]);
        acc[cur] = colors[cur];
        return acc;
    }, colors);
}
function Theme() {
    this.colors = updateColors(theme$1 === 'light' ? lightThemeColors : darkThemeColors);
}
var theme = new Theme();

var BORDER_THICKNESS = 2;
var BORDER_COLOR = theme.colors['--gray-color'];
var HIGHLIGHT_COLOR = theme.colors['--secondary-color'];
var WHITE_COLOR = theme.colors['--background-color'];
var INNERMOST_RATIO = 0.35;
var MINOR_RATIO = 0.49;
var INNER_RATIO = 0.67;
var MAJOR_RATIO = 0.82;
var InnerSemiCircle = function (_a) {
    var backgroundColor = _a.backgroundColor, angleOffset = _a.angleOffset, opacity = _a.opacity;
    return (o$1("div", { style: {
            position: 'absolute',
            transform: "rotate(".concat(angleOffset, "deg)"),
            transformOrigin: 'bottom center',
            height: "".concat(INNER_RATIO * 50, "%"),
            width: "".concat(INNER_RATIO * 100, "%"),
            top: "".concat((1 - INNER_RATIO) * 50, "%"),
            left: "".concat((1 - INNER_RATIO) * 50, "%"),
            borderTopLeftRadius: "".concat(INNER_RATIO * 100, "% ").concat(INNER_RATIO * 200, "%"),
            borderTopRightRadius: "".concat(INNER_RATIO * 100, "% ").concat(INNER_RATIO * 200, "%"),
            backgroundColor: backgroundColor,
            opacity: opacity
        } }));
};
var OuterSemiCircle = function (_a) {
    var backgroundColor = _a.backgroundColor, angleOffset = _a.angleOffset, opacity = _a.opacity;
    return (o$1("div", { style: {
            position: 'absolute',
            top: 0,
            transform: "rotate(".concat(angleOffset, "deg)"),
            transformOrigin: 'bottom center',
            height: '50%',
            width: '100%',
            borderTopLeftRadius: '100% 200%',
            borderTopRightRadius: '100% 200%',
            backgroundColor: backgroundColor,
            opacity: opacity
        } }));
};
var SemiCircleHighlight = function (_a) {
    var result = _a.result, offset = _a.offset;
    var majorKeyIndex = majorKeys.findIndex(function (key) { return key === result; });
    var minorKeyIndex = minorKeys.findIndex(function (key) { return key === result; });
    if (majorKeyIndex >= 0) {
        return (o$1(k$2, { children: [o$1(OuterSemiCircle, { opacity: 0.6, backgroundColor: HIGHLIGHT_COLOR, angleOffset: (majorKeyIndex + offset) * 30 - (90 - 15) }), o$1(OuterSemiCircle, { opacity: 1, backgroundColor: WHITE_COLOR, angleOffset: (majorKeyIndex + offset - 1) * 30 - (90 - 15) }), o$1("div", { style: {
                        position: 'absolute',
                        height: "".concat(INNER_RATIO * 100, "%"),
                        width: "".concat(INNER_RATIO * 100, "%"),
                        top: "".concat((1 - INNER_RATIO) * 50, "%"),
                        left: "".concat((1 - INNER_RATIO) * 50, "%"),
                        borderRadius: '50%',
                        backgroundColor: "".concat(WHITE_COLOR)
                    } })] }));
    }
    else if (minorKeyIndex >= 0) {
        return (o$1(k$2, { children: [o$1(InnerSemiCircle, { opacity: 0.6, backgroundColor: HIGHLIGHT_COLOR, angleOffset: (minorKeyIndex + offset) * 30 - (90 - 15) }), o$1(InnerSemiCircle, { opacity: 1, backgroundColor: WHITE_COLOR, angleOffset: (minorKeyIndex + offset - 1) * 30 - (90 - 15) })] }));
    }
    return null;
};
var CircleOfFifths = /** @class */ (function (_super) {
    __extends(CircleOfFifths, _super);
    function CircleOfFifths() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    CircleOfFifths.prototype.render = function () {
        var offset = majorKeys.indexOf(keyAtTopPosition) * -1;
        return (o$1("div", __assign({ style: { position: 'relative', width: '100%', paddingTop: '100%' } }, { children: [o$1("div", { style: {
                        position: 'absolute',
                        top: 0,
                        borderRadius: '50%',
                        border: "".concat(BORDER_THICKNESS, "px solid ").concat(BORDER_COLOR),
                        height: '100%',
                        width: '100%',
                        backgroundColor: "".concat(WHITE_COLOR)
                    } }), o$1(SemiCircleHighlight, { offset: offset, result: this.props.result }), majorKeys.map(function (_, index) { return (o$1("div", { style: {
                        top: 0,
                        left: "calc(50% - ".concat(BORDER_THICKNESS / 2, "px)"),
                        width: "".concat(BORDER_THICKNESS, "px"),
                        height: '50%',
                        backgroundColor: "".concat(BORDER_COLOR),
                        transform: "rotate(".concat((index + offset) * 30 - 15, "deg)"),
                        transformOrigin: 'bottom center',
                        position: 'absolute'
                    } })); }), o$1("div", { style: {
                        position: 'absolute',
                        borderRadius: '100%',
                        border: "".concat(BORDER_THICKNESS, "px solid ").concat(BORDER_COLOR),
                        height: "".concat(INNERMOST_RATIO * 100, "%"),
                        width: "".concat(INNERMOST_RATIO * 100, "%"),
                        top: "".concat((1 - INNERMOST_RATIO) * 50, "%"),
                        left: "".concat((1 - INNERMOST_RATIO) * 50, "%"),
                        backgroundColor: "".concat(WHITE_COLOR)
                    } }), !this.props.mini &&
                    majorKeys.map(function (major, index) { return (o$1("div", __assign({ style: {
                            top: "".concat((1 - MAJOR_RATIO) * 50, "%"),
                            left: '50%',
                            height: "".concat(MAJOR_RATIO * 50, "%"),
                            width: '0',
                            transform: "rotate(".concat((index + offset) * 30, "deg)"),
                            transformOrigin: 'bottom center',
                            position: 'absolute'
                        } }, { children: o$1("div", __assign({ style: {
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                position: 'relative',
                                left: '-1.5rem',
                                top: '-1.5rem',
                                width: '3rem',
                                height: '3rem',
                                textAlign: 'center',
                                transform: "rotate(".concat(-(index + offset) * 30, "deg)"),
                                fontSize: "".concat(1.1 - Math.sqrt(keysNotation[major].length) * 0.1, "rem"),
                                fontWeight: 'bold'
                            } }, { children: keysNotation[major] })) }))); }), !this.props.mini &&
                    minorKeys.map(function (minor, index) { return (o$1("div", __assign({ style: {
                            top: "".concat((1 - MINOR_RATIO) * 50, "%"),
                            left: '50%',
                            height: "".concat(MINOR_RATIO * 50, "%"),
                            width: '0',
                            transform: "rotate(".concat((index + offset) * 30, "deg)"),
                            transformOrigin: 'bottom center',
                            position: 'absolute'
                        } }, { children: o$1("div", __assign({ style: {
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                position: 'relative',
                                left: '-1rem',
                                top: '-1rem',
                                width: '2rem',
                                height: '2rem',
                                textAlign: 'center',
                                transform: "rotate(".concat(-(index + offset) * 30, "deg)"),
                                fontSize: "".concat(0.9 - Math.sqrt(keysNotation[minor].length) * 0.1, "rem")
                            } }, { children: keysNotation[minor] })) }))); }), o$1("div", { style: {
                        position: 'absolute',
                        top: 0,
                        borderRadius: '50%',
                        border: "".concat(BORDER_THICKNESS, "px solid ").concat(BORDER_COLOR),
                        height: "100%",
                        width: "100%"
                    } }), o$1("div", { style: {
                        position: 'absolute',
                        borderRadius: '100%',
                        border: "".concat(BORDER_THICKNESS, "px solid ").concat(BORDER_COLOR),
                        height: "".concat(INNER_RATIO * 100, "%"),
                        width: "".concat(INNER_RATIO * 100, "%"),
                        top: "".concat((1 - INNER_RATIO) * 50, "%"),
                        left: "".concat((1 - INNER_RATIO) * 50, "%")
                    } })] })));
    };
    CircleOfFifths.defaultProps = {
        mini: false
    };
    return CircleOfFifths;
}(b$2));

var WIDTH = 200;
var HEIGHT = 100;
var LiveDetection = /** @class */ (function (_super) {
    __extends(LiveDetection, _super);
    function LiveDetection() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.audioContext = null;
        _this.recorder = null;
        _this.levelAnalyzer = null;
        _this.keyAnalyzer = null;
        _this.sampleRate = null;
        _this.canvas = _$1();
        _this.canvasContext = null;
        _this.dataArray = null;
        _this.state = {
            connected: false,
            analyzing: false,
            result: null,
            error: null
        };
        _this.drawLevelAnalysis = function () {
            requestAnimationFrame(_this.drawLevelAnalysis);
            _this.levelAnalyzer.getByteTimeDomainData(_this.dataArray);
            _this.canvasContext.fillStyle = theme.colors['--gray-color'];
            _this.canvasContext.fillRect(0, 0, WIDTH, HEIGHT);
            _this.canvasContext.lineWidth = 2;
            _this.canvasContext.strokeStyle = theme.colors['--secondary-color'];
            _this.canvasContext.beginPath();
            var bufferLength = _this.levelAnalyzer.frequencyBinCount;
            var sliceWidth = (WIDTH * 1.0) / bufferLength;
            var x = 0;
            for (var i = 0; i < bufferLength; i++) {
                var v = _this.dataArray[i] / 128.0;
                var y = (v * HEIGHT) / 2;
                if (i === 0) {
                    _this.canvasContext.moveTo(x, y);
                }
                else {
                    _this.canvasContext.lineTo(x, y);
                }
                x += sliceWidth;
            }
            _this.canvasContext.lineTo(WIDTH, HEIGHT / 2);
            _this.canvasContext.stroke();
        };
        _this.routeSound = function () { return __awaiter(_this, void 0, void 0, function () {
            var stream, source, _a, e_1;
            var _this = this;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 3, , 4]);
                        return [4 /*yield*/, requestUserMedia()];
                    case 1:
                        stream = _b.sent();
                        this.audioContext = createAudioContext();
                        source = createAudioSource(this.audioContext, stream);
                        this.sampleRate = getSourceMeta(source).sampleRate;
                        _a = this;
                        return [4 /*yield*/, createRecordingDevice(this.audioContext)];
                    case 2:
                        _a.recorder = _b.sent();
                        this.levelAnalyzer = createAnalyserDevice(this.audioContext);
                        this.dataArray = createDataArrayForAnalyzerDevice(this.levelAnalyzer);
                        this.canvasContext = this.canvas.current.getContext('2d');
                        connectAudioNodes(source, this.recorder);
                        connectAudioNodes(source, this.levelAnalyzer);
                        this.drawLevelAnalysis();
                        this.setState({ connected: true });
                        this.recorder.port.onmessage = function (e) {
                            if (e.data.eventType === 'data') {
                                var audioData = e.data.audioBuffer;
                                _this.keyAnalyzer &&
                                    _this.keyAnalyzer.postMessage({
                                        funcName: 'feedAudioData',
                                        data: [audioData]
                                    });
                            }
                            if (e.data.eventType === 'stop') {
                                _this.keyAnalyzer &&
                                    _this.keyAnalyzer.postMessage({ funcName: 'finalDetection' });
                            }
                        };
                        return [3 /*break*/, 4];
                    case 3:
                        e_1 = _b.sent();
                        this.setState({ error: e_1.message });
                        return [3 /*break*/, 4];
                    case 4: return [2 /*return*/];
                }
            });
        }); };
        _this.connectKeyAnalyzer = function () {
            _this.keyAnalyzer = initializeKeyFinder({
                sampleRate: _this.sampleRate,
                numberOfChannels: 1
            });
            _this.keyAnalyzer.addEventListener('message', function (event) {
                if (event.data.finalResponse) {
                    var result = extractResultFromByteArray(event.data.data);
                    _this.setState({ result: result });
                    _this.keyAnalyzer && _this.keyAnalyzer.terminate();
                }
                else {
                    // Not final response
                    if (event.data.data === 0) {
                        // very first response
                        console.log('Analyzer is initialized');
                        _this.setState({ analyzing: true });
                    }
                    else {
                        // not first response
                        var result = extractResultFromByteArray(event.data.data);
                        _this.setState({ result: result });
                    }
                }
            });
        };
        _this.startRecording = function () {
            if (!_this.recorder || !_this.audioContext)
                return;
            _this.connectKeyAnalyzer();
            var contextTime = _this.audioContext.getOutputTimestamp().contextTime;
            _this.recorder.parameters
                .get('isRecording')
                .setValueAtTime(1, contextTime + 0.1);
            _this.setState({ result: '...' });
        };
        _this.stopRecording = function () {
            if (!_this.recorder || !_this.audioContext)
                return;
            _this.setState({ analyzing: false });
            var contextTime = _this.audioContext.getOutputTimestamp().contextTime;
            _this.recorder.parameters
                .get('isRecording')
                .setValueAtTime(0, contextTime + 0.1);
        };
        return _this;
    }
    LiveDetection.prototype.componentDidMount = function () {
        document.title = 'keyfinder | Key Finder for Live Audio';
        document
            .querySelector('meta[name="description"]')
            .setAttribute('content', 'A web application to find the musical key (root note) of a song from the live audio feed. Analyze audio from your microphone or audio stream routed from your sound card to find the root note right in your browser.');
    };
    LiveDetection.prototype.componentWillUnmount = function () {
        this.keyAnalyzer && this.keyAnalyzer.terminate();
    };
    LiveDetection.prototype.render = function (_a, _b) {
        var connected = _b.connected, analyzing = _b.analyzing, result = _b.result, error = _b.error;
        return (o$1("div", __assign({ "class": "live-detection-page" }, { children: [error && o$1("h1", { children: error }), o$1("main", __assign({ "class": "live-detection__container" }, { children: [o$1("div", __assign({ style: { display: 'flex', flexDirection: 'column' } }, { children: [o$1("header", { children: o$1("h1", __assign({ style: { marginTop: 0 } }, { children: "Live Key Detection" })) }), o$1("div", { children: [o$1("div", __assign({ style: { paddingBottom: '2rem' } }, { children: o$1("input", { type: "button", onClick: this.routeSound, value: connected
                                                    ? 'Key detection engine running'
                                                    : 'Route sound to key detection engine', disabled: connected }) })), o$1("div", __assign({ style: { paddingBottom: '2rem' } }, { children: [o$1("input", { type: "button", onClick: this.startRecording, value: "Start Key Detection", disabled: !connected || analyzing, style: { marginBottom: '0.5rem' } }), o$1("input", { type: "button", onClick: this.stopRecording, value: "End Key Detection", disabled: !analyzing })] })), o$1("div", { children: o$1("canvas", { width: WIDTH, height: HEIGHT, ref: this.canvas, style: { width: WIDTH, height: HEIGHT } }) }), o$1("div", __assign({ style: { height: '2rem' } }, { children: result &&
                                                "".concat(analyzing ? 'Progressive' : 'Final', " Result: ").concat(keysNotation[result] || result) }))] })] })), o$1("div", __assign({ "class": "live-detection__circle-of-fifths" }, { children: o$1(CircleOfFifths, { result: result }) }))] }))] })));
    };
    return LiveDetection;
}(b$2));

var AudioFileKeyDetection$1 = /** @class */ (function (_super) {
    __extends(AudioFileKeyDetection, _super);
    function AudioFileKeyDetection() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.worker = null;
        _this.terminated = false;
        _this.state = {
            analysisStart: null,
            analysisDuration: null,
            currentSegment: null,
            maxSegments: null,
            analyzing: false,
            result: null
        };
        _this.advanceSegmentCount = function () {
            _this.setState(function (_a) {
                var currentSegment = _a.currentSegment;
                return ({
                    currentSegment: currentSegment + 1
                });
            });
        };
        _this.postAudioSegmentAtOffset = function (worker, channelData, sampleRate, numberOfChannels, offset) {
            var segment = zipChannelsAtOffset(channelData, offset, sampleRate, numberOfChannels);
            worker.postMessage({ funcName: 'feedAudioData', data: [segment] });
        };
        _this.handleAudioFile = function (buffer) {
            if (_this.terminated)
                return;
            var sampleRate = buffer.sampleRate;
            var numberOfChannels = buffer.numberOfChannels;
            var channelData = [];
            for (var i = 0; i < numberOfChannels; i += 1) {
                channelData.push(buffer.getChannelData(i));
            }
            _this.setState({
                analyzing: true,
                analysisStart: performance.now(),
                analysisDuration: null
            });
            _this.worker = initializeKeyFinder({
                sampleRate: sampleRate,
                numberOfChannels: numberOfChannels
            });
            var segmentCounts = Math.floor(channelData[0].length / sampleRate);
            _this.setState({ maxSegments: segmentCounts, currentSegment: 0 });
            _this.worker.addEventListener('message', function (event) {
                if (event.data.finalResponse) {
                    var result_1 = extractResultFromByteArray(event.data.data);
                    _this.setState(function (oldState) { return ({
                        result: result_1,
                        analysisDuration: performance.now() - oldState.analysisStart,
                        analyzing: false
                    }); });
                    _this.props.updateResult(_this.props.fileItem.id, result_1);
                    _this.worker.terminate();
                    _this.worker = null;
                }
                else {
                    // Not final response
                    if (event.data.data === 0) {
                        // very first response
                        _this.postAudioSegmentAtOffset(_this.worker, channelData, sampleRate, numberOfChannels, 0);
                        _this.advanceSegmentCount();
                    }
                    else {
                        // not first response
                        var result = extractResultFromByteArray(event.data.data);
                        _this.setState({ result: result });
                        if (_this.state.currentSegment < segmentCounts) {
                            var offset = _this.state.currentSegment * sampleRate;
                            _this.postAudioSegmentAtOffset(_this.worker, channelData, sampleRate, numberOfChannels, offset);
                            _this.advanceSegmentCount();
                        }
                        else {
                            // no more segments
                            _this.worker.postMessage({ funcName: 'finalDetection' });
                        }
                    }
                }
            });
        };
        _this.handleFileLoad = function (event) { return __awaiter(_this, void 0, void 0, function () {
            var context, digest, hashArray, hashHex;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        context = createAudioContext();
                        return [4 /*yield*/, crypto.subtle.digest('SHA-256', event.target.result)];
                    case 1:
                        digest = _a.sent();
                        hashArray = Array.from(new Uint8Array(digest));
                        hashHex = hashArray
                            .map(function (b) { return b.toString(16).padStart(2, '0'); })
                            .join('');
                        this.props.updateDigest(this.props.fileItem.id, hashHex);
                        context.decodeAudioData(event.target.result, this.handleAudioFile);
                        return [2 /*return*/];
                }
            });
        }); };
        return _this;
    }
    AudioFileKeyDetection.prototype.componentDidMount = function () {
        if (this.props.fileItem.canProcess) {
            var reader = new FileReader();
            reader.onload = this.handleFileLoad;
            reader.readAsArrayBuffer(this.props.fileItem.file);
        }
    };
    AudioFileKeyDetection.prototype.componentDidUpdate = function (prevProps) {
        if (prevProps.fileItem.canProcess === false &&
            this.props.fileItem.canProcess === true) {
            var reader = new FileReader();
            reader.onload = this.handleFileLoad;
            reader.readAsArrayBuffer(this.props.fileItem.file);
        }
    };
    AudioFileKeyDetection.prototype.componentWillUnmount = function () {
        this.terminated = true;
        this.worker && this.worker.terminate();
    };
    AudioFileKeyDetection.prototype.render = function (_a, _b) {
        var fileItem = _a.fileItem;
        var currentSegment = _b.currentSegment, maxSegments = _b.maxSegments; _b.analyzing; var result = _b.result, analysisDuration = _b.analysisDuration;
        return (o$1("div", __assign({ "class": "file-item__container" }, { children: [o$1("div", __assign({ "class": "file-item__song-name" }, { children: fileItem.file.name })), o$1("div", __assign({ "class": "file-item__result-container" }, { children: [o$1("div", __assign({ "class": "file-item__result-text" }, { children: result && keysNotation[result] && "".concat(keysNotation[result]) })), o$1("div", __assign({ "class": "file-item__circle" }, { children: o$1(CircleOfFifths, { mini: true, result: result }) }))] })), o$1("div", __assign({ "class": "file-item__progress-indicator" }, { children: [o$1("progress", { value: currentSegment, max: maxSegments }), result &&
                            analysisDuration &&
                            "".concat((analysisDuration / 1000).toFixed(1), " s")] }))] })));
    };
    return AudioFileKeyDetection;
}(b$2));

// Unique ID creation requires a high quality random # generator. In the browser we therefore
// require the crypto API and do not support built-in fallback to lower quality random number
// generators (like Math.random()).
let getRandomValues;
const rnds8 = new Uint8Array(16);
function rng() {
  // lazy load so that environments that need to polyfill have a chance to do so
  if (!getRandomValues) {
    // getRandomValues needs to be invoked in a context where "this" is a Crypto implementation.
    getRandomValues = typeof crypto !== 'undefined' && crypto.getRandomValues && crypto.getRandomValues.bind(crypto);

    if (!getRandomValues) {
      throw new Error('crypto.getRandomValues() not supported. See https://github.com/uuidjs/uuid#getrandomvalues-not-supported');
    }
  }

  return getRandomValues(rnds8);
}

/**
 * Convert array of 16 byte values to UUID string format of the form:
 * XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
 */

const byteToHex = [];

for (let i = 0; i < 256; ++i) {
  byteToHex.push((i + 0x100).toString(16).slice(1));
}

function unsafeStringify(arr, offset = 0) {
  // Note: Be careful editing this code!  It's been tuned for performance
  // and works in ways you may not expect. See https://github.com/uuidjs/uuid/pull/434
  return (byteToHex[arr[offset + 0]] + byteToHex[arr[offset + 1]] + byteToHex[arr[offset + 2]] + byteToHex[arr[offset + 3]] + '-' + byteToHex[arr[offset + 4]] + byteToHex[arr[offset + 5]] + '-' + byteToHex[arr[offset + 6]] + byteToHex[arr[offset + 7]] + '-' + byteToHex[arr[offset + 8]] + byteToHex[arr[offset + 9]] + '-' + byteToHex[arr[offset + 10]] + byteToHex[arr[offset + 11]] + byteToHex[arr[offset + 12]] + byteToHex[arr[offset + 13]] + byteToHex[arr[offset + 14]] + byteToHex[arr[offset + 15]]).toLowerCase();
}

const randomUUID = typeof crypto !== 'undefined' && crypto.randomUUID && crypto.randomUUID.bind(crypto);
var native = {
  randomUUID
};

function v4(options, buf, offset) {
  if (native.randomUUID && !buf && !options) {
    return native.randomUUID();
  }

  options = options || {};
  const rnds = options.random || (options.rng || rng)(); // Per 4.4, set bits for version and `clock_seq_hi_and_reserved`

  rnds[6] = rnds[6] & 0x0f | 0x40;
  rnds[8] = rnds[8] & 0x3f | 0x80; // Copy bytes to buffer, if provided

  if (buf) {
    offset = offset || 0;

    for (let i = 0; i < 16; ++i) {
      buf[offset + i] = rnds[i];
    }

    return buf;
  }

  return unsafeStringify(rnds);
}

var AudioFileKeyDetection = /** @class */ (function (_super) {
    __extends(AudioFileKeyDetection, _super);
    function AudioFileKeyDetection() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.ref = _$1();
        _this.state = {
            files: []
        };
        _this.handleFileInput = function (_a) {
            var target = _a.target;
            var fileList = target.files;
            _this.setState(function (_a) {
                var files = _a.files;
                var availableThreads = files.reduce(function (acc, cur) {
                    if (cur.canProcess && !cur.result)
                        return acc - 1;
                    return acc;
                }, numberOfThreads);
                for (var fileIdx = 0; fileIdx < fileList.length; fileIdx += 1) {
                    var canProcess = false;
                    if (availableThreads > 0) {
                        canProcess = true;
                        availableThreads -= 1;
                    }
                    var id = v4();
                    files.push({
                        id: id,
                        canProcess: canProcess,
                        file: fileList[fileIdx],
                        result: null,
                        digest: null
                    });
                }
                _this.ref.current.value = null;
                return { files: files };
            });
        };
        _this.updateDigest = function (uuid, digest) {
            _this.setState(function (_a) {
                var files = _a.files;
                var newFiles = files.map(function (file) {
                    if (file.id === uuid)
                        return __assign(__assign({}, file), { uuid: uuid });
                    return file;
                });
                return { files: newFiles };
            });
        };
        _this.updateResult = function (uuid, result) {
            _this.setState(function (_a) {
                var files = _a.files;
                var availableThreads = 1;
                var newFiles = files.map(function (file) {
                    if (file.id === uuid)
                        return __assign(__assign({}, file), { result: result });
                    if (file.canProcess === false && availableThreads > 0) {
                        availableThreads -= 1;
                        return __assign(__assign({}, file), { canProcess: true });
                    }
                    return file;
                });
                return { files: newFiles };
            });
        };
        return _this;
    }
    AudioFileKeyDetection.prototype.componentDidMount = function () {
        document.title = 'keyfinder | Key Finder for Audio Files';
        document
            .querySelector('meta[name="description"]')
            .setAttribute('content', 'A web application to find the musical key (root note) of an audio file. Song will be analyzed right in your browser. Select the audio file from your computer to find the root note.');
    };
    AudioFileKeyDetection.prototype.render = function (_a, _b) {
        var _this = this;
        var files = _b.files;
        return (o$1("main", __assign({ "class": "audio-file-key-detection-page" }, { children: [o$1("header", { children: o$1("h1", { children: "Audio File Key Detection" }) }), o$1("div", __assign({ style: { paddingTop: '1rem' } }, { children: [o$1("p", __assign({ style: { fontSize: '0.6rem' } }, { children: [numberOfThreads === 1
                                    ? 'No parallel processes. '
                                    : "Using ".concat(numberOfThreads, " parallel processes. "), o$1(E, __assign({ href: "/settings" }, { children: "[settings]" }))] })), o$1("div", __assign({ style: { marginBottom: '2rem' } }, { children: [o$1("label", __assign({ "for": "load-a-track", style: { paddingRight: '1rem' } }, { children: ["Load a track:", ' '] })), o$1("input", { ref: this.ref, id: "load-a-track", type: "file", accept: "audio/*", multiple: true, onChange: this.handleFileInput })] })), files.map(function (fileItem) { return (o$1(AudioFileKeyDetection$1, { fileItem: fileItem, updateDigest: _this.updateDigest, updateResult: _this.updateResult }, fileItem.id)); })] }))] })));
    };
    return AudioFileKeyDetection;
}(b$2));

var Settings = /** @class */ (function (_super) {
    __extends(Settings, _super);
    function Settings() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.state = {
            keysNotation: keysNotation,
            theme: theme$1,
            keyAtTopPosition: keyAtTopPosition,
            numberOfThreads: numberOfThreads
        };
        _this.handleSave = function (e) {
            e.preventDefault();
            try {
                localStorage.setItem('customSettings', JSON.stringify(_this.state));
                location.reload();
            }
            catch (e) {
                console.error('Can not use local storage', e);
            }
        };
        _this.onInputNewNotation = function (e) {
            var _a;
            var _b = e.target, value = _b.value, id = _b.id;
            _this.setState({
                keysNotation: __assign(__assign({}, _this.state.keysNotation), (_a = {}, _a[id] = value, _a))
            });
        };
        _this.onInput = function (e) {
            var _a;
            var _b = e.target, value = _b.value, id = _b.id;
            _this.setState((_a = {},
                _a[id] = value,
                _a));
        };
        _this.onChange = function (e) {
            var _a;
            var _b = e.target, value = _b.value, name = _b.name;
            _this.setState((_a = {}, _a[name] = value, _a));
        };
        _this.handleReset = function () {
            localStorage.clear();
            location.reload();
        };
        return _this;
    }
    Settings.prototype.componentDidMount = function () {
        document.title = 'keyfinder | Settings for Key Finder Application';
        document
            .querySelector('meta[name="description"]')
            .setAttribute('content', 'Adjust the settings for the musical key finder application. You can modify the notation used to visualize the circle of fifths.');
    };
    Settings.prototype.render = function () {
        var _this = this;
        return (o$1("main", __assign({ "class": "settings-page" }, { children: [o$1("header", { children: o$1("h1", { children: "Settings" }) }), o$1("div", __assign({ "class": "settings-container" }, { children: [o$1("p", { children: 'Custom settings are stored locally. Change values as you desire and click on the save button at the bottom.' }), o$1("form", __assign({ onSubmit: this.handleSave }, { children: [o$1("h2", { children: "General" }), o$1("h3", { children: "Alternative Notation" }), o$1("p", { children: 'Update default notation by modifying respective fields. Use following characters: a-z, A-Z, 0-9, ♭, ♯. No spaces in the beginning or the end.' }), o$1("div", __assign({ "class": "settings-container__notation-fields" }, { children: [o$1("div", __assign({ "class": "settings-container__notation-fields-column" }, { children: majorKeys.map(function (major) { return (o$1("div", __assign({ "class": "settings-container__notation-field" }, { children: [o$1("label", __assign({ "for": major }, { children: major })), o$1("input", { onInput: _this.onInputNewNotation, id: major, value: _this.state.keysNotation[major], pattern: "[\\w\u266D\u266F]|[\\w\u266D\u266F][\\w\\s\u266D\u266F]*[\\w\u266D\u266F]" })] }))); }) })), o$1("div", __assign({ "class": "settings-container__notation-fields-column" }, { children: minorKeys.map(function (minor) { return (o$1("div", __assign({ "class": "settings-container__notation-field" }, { children: [o$1("label", __assign({ "for": minor }, { children: minor })), o$1("input", { onInput: _this.onInputNewNotation, id: minor, value: _this.state.keysNotation[minor], pattern: "[\\w\u266D\u266F]|[\\w\u266D\u266F][\\w\\s\u266D\u266F]*[\\w\u266D\u266F]" })] }))); }) }))] })), o$1("h3", { children: "Theme" }), o$1("div", __assign({ "class": "settings-container__theme-fields" }, { children: [o$1("div", { children: [o$1("input", { type: "radio", id: "light", name: "theme", value: "light", onChange: this.onChange, checked: this.state.theme === 'light' }), o$1("label", __assign({ "for": "light" }, { children: "light" }))] }), o$1("div", { children: [o$1("input", { type: "radio", id: "dark", name: "theme", value: "dark", onChange: this.onChange, checked: this.state.theme === 'dark' }), o$1("label", __assign({ "for": "dark" }, { children: "dark" }))] })] })), o$1("h2", { children: "Live Detection" }), o$1("p", { children: "Some notations orient circle of fifths differently. Select the note in the 12 o'clock position to adjust how circle of fifths is visualized." }), o$1("div", __assign({ "class": "settings-container__key-at-top-field" }, { children: [o$1("label", __assign({ "for": "keyAtTopPosition" }, { children: o$1("h3", { children: "Key at top position" }) })), o$1("select", __assign({ name: "keyAtTopPosition", id: "keyAtTopPosition", onChange: this.onChange }, { children: o$1(k$2, { children: majorKeys.map(function (key) { return (o$1("option", __assign({ value: key, selected: _this.state.keyAtTopPosition === key }, { children: key }))); }) }) }))] })), o$1("h2", { children: "File Analysis" }), o$1("p", { children: 'While analyzing files, the application spawns multiple workers. Set the maximum number of workers to be run at the same time.' }), o$1("div", __assign({ "class": "settings-container__processes-field" }, { children: [o$1("label", __assign({ "for": "numberOfThreads" }, { children: o$1("h3", { children: "Parallel Processes" }) })), o$1("input", { type: "number", id: "numberOfThreads", name: "numberOfThreads", min: "1", onInput: this.onInput, value: this.state.numberOfThreads })] })), this.state.numberOfThreads > maxNumberOfThreads && (o$1("p", __assign({ "class": "settings-container--danger" }, { children: "According to your browser, your machine has ".concat(maxNumberOfThreads, " processors available. Spawing more threads than that will slow down your computer.") }))), o$1("button", __assign({ "class": "settings-container__save-button", type: "submit" }, { children: "SAVE" }))] })), o$1("div", __assign({ "class": "settings-container__reset-section" }, { children: [o$1("h3", __assign({ "class": "settings-container--danger" }, { children: "DANGER" })), o$1("button", __assign({ onClick: this.handleReset }, { children: "delete custom settings" }))] }))] }))] })));
    };
    return Settings;
}(b$2));

var ProjectInfoLink = function () { return (o$1("a", __assign({ href: "https://doga.dev/projects/web-key-finder", target: "_blank", rel: "noopener noreferrer" }, { children: "doga.dev/web-key-finder" }))); };
var GithubLink = function () { return (o$1("a", __assign({ href: "https://github.com/dogayuksel/webKeyFinder", target: "_blank", rel: "noopener noreferrer" }, { children: "github" }))); };
var EmailAddress = function () { return (o$1("a", __assign({ href: "mailto:hello@doga.dev" }, { children: "hello[at]doga.dev" }))); };
var About = /** @class */ (function (_super) {
    __extends(About, _super);
    function About() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    About.prototype.componentDidMount = function () {
        document.title = 'keyfinder | More about the Key Finder Application';
        document
            .querySelector('meta[name="description"]')
            .setAttribute('content', 'Find out more about how the Key Finder Application application works. Access the source code to run it yourself.');
    };
    About.prototype.render = function () {
        return (o$1("main", __assign({ "class": "about-page" }, { children: [o$1("header", { children: o$1("h1", { children: "About" }) }), o$1("div", __assign({ "class": "about-page__links" }, { children: [o$1("p", { children: ["More info about the project at ", o$1(ProjectInfoLink, {}), "."] }), o$1("p", { children: ["Source code is available on ", o$1(GithubLink, {}), "."] }), o$1("p", { children: ["If you do not have a github account, shoot me an email at", ' ', o$1(EmailAddress, {}), " to request it."] })] }))] })));
    };
    return About;
}(b$2));

var App = /** @class */ (function (_super) {
    __extends(App, _super);
    function App() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    App.prototype.render = function () {
        return (o$1(k$2, { children: [o$1("div", __assign({ "class": "top-bar" }, { children: [o$1("div", __assign({ "class": "app-logo" }, { children: o$1(E, __assign({ href: "/" }, { children: "keyfinder" })) })), o$1(App$1, {})] })), o$1("div", __assign({ "class": "app-wrapper" }, { children: o$1(D, { children: [o$1(LiveDetection, { "default": true }), o$1(AudioFileKeyDetection, { path: "/file" }), o$1(Settings, { path: "/settings" }), o$1(About, { path: "/about" })] }) }))] }));
    };
    return App;
}(b$2));

D$1(o$1(App, {}), document.body);
