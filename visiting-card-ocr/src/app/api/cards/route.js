import clientPromise from '../../../utils/db';

export async function POST(req) {
  try {
    const client = await clientPromise;
    const db = client.db('visiting-cards');
    const body = await req.json();

    // Insert the JSON data into 'cards' collection
    const result = await db.collection('cards').insertOne(body);

    return Response.json({ success: true, insertedId: result.insertedId });
  } catch (error) {
    console.error('❌ Failed to insert into MongoDB:', error);
    return Response.json({ success: false, error: 'Failed to save data' }, { status: 500 });
  }
}