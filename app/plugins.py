from litestar_granian import GranianPlugin
from litestar_vite import ViteConfig, VitePlugin
from litestar_vite.inertia import InertiaConfig, InertiaPlugin

granian = GranianPlugin()
vite = VitePlugin(ViteConfig())
inertia = InertiaPlugin(InertiaConfig())
