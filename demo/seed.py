"""Create the demo table: bilingual help-centre articles with vectors from an "old" model.

Runs inside the demo container. The vectors come from vecshift's offline hashing model,
so the demo needs no API key and sends nothing anywhere.
"""

from __future__ import annotations

import asyncio
import os
import random

import psycopg

from vecshift.embeddings import create_embedder, parse_spec

OLD_MODEL = "hash/64"
ROWS_PER_TOPIC = 40

# (English templates, Arabic templates) per topic. {city}, {n} and {plan} vary per row.
TOPICS: list[tuple[list[str], list[str]]] = [
    (
        [
            "How do I reset my password if I forgot it?",
            "The password reset email never arrived in {city}.",
            "Reset link expired after {n} minutes, how can I get a new one?",
            "Can I change my password from the mobile app?",
        ],
        [
            "كيف أعيد تعيين كلمة المرور إذا نسيتها؟",
            "لم تصلني رسالة إعادة تعيين كلمة المرور في {city_ar}.",
            "انتهت صلاحية رابط إعادة التعيين بعد {n} دقيقة، كيف أحصل على رابط جديد؟",
            "هل يمكنني تغيير كلمة المرور من تطبيق الجوال؟",
        ],
    ),
    (
        [
            "My card was charged twice for the {plan} plan.",
            "How do I get a refund for an order I cancelled?",
            "Refunds take {n} business days to reach your bank account.",
            "Why was I billed after cancelling my subscription?",
        ],
        [
            "تم خصم المبلغ مرتين من بطاقتي لخطة {plan_ar}.",
            "كيف أسترد المبلغ لطلب قمت بإلغائه؟",
            "يستغرق وصول المبلغ المسترد إلى حسابك البنكي {n} أيام عمل.",
            "لماذا تم خصم مبلغ بعد إلغاء اشتراكي؟",
        ],
    ),
    (
        [
            "Where is my order? It was supposed to arrive in {city} yesterday.",
            "Track your shipment with the number in the confirmation email.",
            "Delivery to {city} usually takes {n} days.",
            "The courier marked my package as delivered but I never received it.",
        ],
        [
            "أين طلبي؟ كان من المفترض أن يصل إلى {city_ar} أمس.",
            "تتبع شحنتك باستخدام الرقم الموجود في رسالة التأكيد.",
            "يستغرق التوصيل إلى {city_ar} عادة {n} أيام.",
            "سجل المندوب أن الطرد تم تسليمه لكنني لم أستلمه.",
        ],
    ),
    (
        [
            "How do I upgrade from the {plan} plan?",
            "What is included in the {plan} plan?",
            "Can I downgrade my plan at the end of the month?",
            "Plans can be changed at any time from the billing page.",
        ],
        [
            "كيف أرقي اشتراكي من خطة {plan_ar}؟",
            "ما الذي تتضمنه خطة {plan_ar}؟",
            "هل يمكنني تخفيض خطتي في نهاية الشهر؟",
            "يمكن تغيير الخطة في أي وقت من صفحة الفواتير.",
        ],
    ),
    (
        [
            "Turn on two-factor authentication to protect your account.",
            "I lost the phone I use for two-factor codes.",
            "Someone in {city} logged into my account, what should I do?",
            "We noticed a new sign-in from {city}.",
        ],
        [
            "فعّل المصادقة الثنائية لحماية حسابك.",
            "فقدت الهاتف الذي أستخدمه لرموز التحقق الثنائي.",
            "قام شخص في {city_ar} بتسجيل الدخول إلى حسابي، ماذا أفعل؟",
            "لاحظنا تسجيل دخول جديد من {city_ar}.",
        ],
    ),
    (
        [
            "The app crashes when I open the camera.",
            "The app is slow after the latest update on Android.",
            "Clear the app cache and restart your phone.",
            "Version {n}.2 fixes the crash on startup.",
        ],
        [
            "يتوقف التطبيق عن العمل عند فتح الكاميرا.",
            "التطبيق بطيء بعد آخر تحديث على أندرويد.",
            "امسح ذاكرة التخزين المؤقت للتطبيق وأعد تشغيل هاتفك.",
            "الإصدار {n}.2 يصلح مشكلة التوقف عند التشغيل.",
        ],
    ),
    (
        [
            "How do I export my data as a CSV file?",
            "Exports with more than {n} thousand rows are emailed to you.",
            "Can I schedule a weekly export of my reports?",
            "Download all your invoices from the account page.",
        ],
        [
            "كيف أصدّر بياناتي كملف CSV؟",
            "يتم إرسال ملفات التصدير التي تتجاوز {n} ألف صف إلى بريدك الإلكتروني.",
            "هل يمكنني جدولة تصدير أسبوعي لتقاريري؟",
            "نزّل جميع فواتيرك من صفحة الحساب.",
        ],
    ),
    (
        [
            "Our office in {city} is open from 9 to {n} on weekdays.",
            "Contact support by chat or phone.",
            "Support is closed on public holidays.",
            "How long does it take to get a reply from support?",
        ],
        [
            "مكتبنا في {city_ar} مفتوح من التاسعة حتى {n} في أيام الأسبوع.",
            "تواصل مع الدعم عبر المحادثة أو الهاتف.",
            "الدعم مغلق في العطلات الرسمية.",
            "كم من الوقت يستغرق الحصول على رد من الدعم؟",
        ],
    ),
    (
        [
            "Invite your team members from the settings page.",
            "Each {plan} workspace can have up to {n} members.",
            "How do I remove someone from my team?",
            "Admins can change a member's role at any time.",
        ],
        [
            "ادعُ أعضاء فريقك من صفحة الإعدادات.",
            "يمكن أن تضم كل مساحة عمل بخطة {plan_ar} حتى {n} عضوًا.",
            "كيف أزيل شخصًا من فريقي؟",
            "يمكن للمسؤولين تغيير دور العضو في أي وقت.",
        ],
    ),
    (
        [
            "Connect your store to the payment gateway in {n} steps.",
            "The API returns error 429 when you send too many requests.",
            "Webhooks are retried for up to {n} hours.",
            "Generate an API key from the developer settings.",
        ],
        [
            "اربط متجرك ببوابة الدفع في {n} خطوات.",
            "تُرجع الواجهة البرمجية الخطأ 429 عند إرسال طلبات كثيرة.",
            "تتم إعادة محاولة إرسال الإشعارات لمدة تصل إلى {n} ساعة.",
            "أنشئ مفتاح API من إعدادات المطورين.",
        ],
    ),
]

# A closing line per row, so articles don't repeat each other.
CLOSINGS = (
    [
        "Thanks for your help.",
        "This is urgent for our team.",
        "I'm on the {plan} plan.",
        "I'm writing from {city}.",
        "This started {n} days ago.",
        "Order number {order}.",
    ],
    [
        "شكرًا لمساعدتكم.",
        "الأمر عاجل لفريقنا.",
        "أنا مشترك في خطة {plan_ar}.",
        "أكتب إليكم من {city_ar}.",
        "بدأت المشكلة منذ {n} أيام.",
        "رقم الطلب {order}.",
    ],
)

CITIES = [
    ("Riyadh", "الرياض"),
    ("Cairo", "القاهرة"),
    ("Dubai", "دبي"),
    ("Amman", "عمّان"),
    ("Casablanca", "الدار البيضاء"),
    ("London", "لندن"),
    ("Berlin", "برلين"),
    ("Doha", "الدوحة"),
]
PLANS = [("Basic", "الأساسية"), ("Pro", "الاحترافية"), ("Business", "الأعمال")]


def articles() -> list[tuple[str, str, str]]:
    """(text, language, topic) rows, the same on every run."""
    rng = random.Random(7)
    rows = []
    for topic, (english, arabic) in enumerate(TOPICS):
        for i in range(ROWS_PER_TOPIC):
            lang = "ar" if i % 2 else "en"
            templates = arabic if lang == "ar" else english
            city, city_ar = rng.choice(CITIES)
            plan, plan_ar = rng.choice(PLANS)
            first, second = rng.sample(templates, 2)
            closing = rng.choice(CLOSINGS[lang == "ar"])
            text = f"{first} {second} {closing}".format(
                city=city,
                city_ar=city_ar,
                plan=plan,
                plan_ar=plan_ar,
                n=rng.randint(2, 30),
                order=rng.randint(10_000, 99_999),
            )
            rows.append((text, lang, f"topic-{topic + 1}"))
    return rows


async def main() -> None:
    rows = articles()
    embedder = create_embedder(parse_spec(OLD_MODEL))
    vectors = await embedder.embed([text for text, _, _ in rows])
    with psycopg.connect(os.environ["VECSHIFT_DSN"], autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute(
            "CREATE TABLE documents (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            "body text NOT NULL, lang text NOT NULL, topic text NOT NULL, "
            "embedding vector(64))"
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (body, lang, topic, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                [
                    (text, lang, topic, "[" + ",".join(map(str, vector)) + "]")
                    for (text, lang, topic), vector in zip(rows, vectors, strict=True)
                ],
            )
        conn.execute(
            "CREATE INDEX documents_embedding_idx ON documents "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute("ANALYZE documents")
    print(
        f"Created table documents: {len(rows)} articles in English and Arabic, "
        f"embedded with {OLD_MODEL}."
    )


if __name__ == "__main__":
    asyncio.run(main())
