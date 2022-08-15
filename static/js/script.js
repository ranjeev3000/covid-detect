'use static';
function _(ele){
    return document.getElementById(ele);
}

console.log("JS loaded");

// _("photo").src = "../static/images/hog_COVID-1.png";

inputImage = function(){
    const ip = _("xrayimg").value;
    const ipName = ip.split("\\").pop();
    console.log(ipName);
    console.log("function activated")
}

if(_("inputPath").textContent!=='')
{
    _("imageshow").classList.remove("d-none");
    _("photo").classList.remove("d-none");
};


// _("xraying").onchange = inputImage;